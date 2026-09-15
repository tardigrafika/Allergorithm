"""
Partial fine-tuning ESM-2 650M (poslednji transformer blok, ili poslednja N)
ZAJEDNO sa MLP glavom, u ISTOM optimizer koraku, unutar SVAKOG LOCO fold-a --
korisnicki zahtev, 2026-09-12: "ne zelim LoRA za ovaj eksperiment... Zelim
da ESM-2 ostane deo computation graph-a i da se azuriraju originalni
parametri poslednjeg (ili poslednja 2) transformer bloka zajedno sa MLP-om
u istom trening loop-u."

RAZLIKA od ml/lora_finetune_esm_1548.py (prethodni pokusaj, odbacen): taj
skript trenira LoRA adaptere ODVOJENO (triplet loss, jedan train/test
split), pa SNIMA rezultujuce embeddinge kao pickle -- MLP koji ih kasnije
koristi (ml/loco_mlp_hadamard_lora650m_1548.py) NIKAD ne vidi ESM-2
gradijente (potvrdjeno citanjem koda: torch.from_numpy(embedding_matrix)
nema autograd istoriju -- dva odvojena treninga, ne end-to-end). OVDE
sekvence prolaze kroz ESM-2 UNUTAR MLP trening petlje: hadamard(embA, embB)
je diferencijabilna funkcija LIVE ESM-2 izlaza, loss.backward() ide kroz
MLP glavu I kroz poslednji ESM-2 blok, JEDAN optimizer.step() poziv azurira
oboje (dve LR grupe: MLP 1e-2, ESM 2e-5).

Mehanizam potvrdjen empirijski pre pisanja ovog skripta (CPU, jedan
forward/backward na 2 kratke sekvence, van ovog fajla): sa samo
encoder.layer[-1] trenabilnim, gradijenti stizu TACNO tamo (19.7M/651M =
3.02% parametara) i nigde drugde -- zamrznuti parametri bit-identicni pre/
posle optimizer.step(), trenabilni se menjaju. Videti PASS ispis u sesiji.

Isti LOCO fold-ovi/skup parova kao zvanicni frozen baseline
(output/loco_mlp_hadamard_1548_summary.txt, 44 folda, MICRO MRR=0.1209):
gold_pairs/all_ids/id_to_index se uzimaju preko load_dataset() na FROZEN
embeddings.pkl SAMO da bi candidate universe (1535 proteina) i redosled
bili identicni -- ne kao izvor embeddinga za trening. Sekvence za live
embedovanje dolaze iz output/clean_allergens.csv (isto kao
ml/lora_finetune_esm_1548.py).

Fold-nezavisnost (plan §9 -- held-out klaster ne sme uticati na ESM-2
azuriranja): model se UCITAVA JEDNOM (ustedi vreme -- 44x ucitavanje 650M
sa diska bi bilo skupo), ali se TRENABILNI blok(ovi) EKSPLICITNO vracaju na
sacuvano pretrained stanje pre svakog fold-a (reset_trainable_block()) --
matematicki identicno ponovnom ucitavanju celog modela, mnogo brze. MLP
glava se pravi iz nule svaki fold (kao i svuda drugde u projektu).

JEDINA promenljiva u ovom eksperimentu je frozen vs fine-tuned ESM-2 --
korisnicki zahtev, 2026-09-12 (drugi put, eksplicitna ispravka): "Ne zelim
novu implementaciju MLP pipeline-a niti promenu trening procedure... Zadrzi
identican pipeline kao u postojecem MLP(hadamard) LOCO eksperimentu... Ne
uvodi LoRA niti menjaj hiperparametre da bi eksperiment bio jeftiniji."

Zato su SVI MLP-strana hiperparametri sada BUKVALNO identicni
ml/loco_mlp_hadamard_1548.py (MLP_HADAMARD_PARAMS) i sample_negatives
pozivu tamo -- NEG_PER_POS=10, MAX_EPOCHS=300, PATIENCE=20,
VAL_FRACTION=0.15, batch_size=64 (parova, ne sekvenci), hidden_dims=[32],
dropout=[0.3], MLP_LR=1e-2, MLP_L2_LAMBDA=1e-3, MAX_LENGTH=1022 svuda (isto
sto i embeddings/generate_embedidngs.py MAX_LENGTH -- originalni frozen
embeddinzi NISU kraceni, pa ni ovde ne kratimo trening sekvence). Redosled
parova u pairs_all (pozitivni pa negativni, u datom redosledu) je BUKVALNO
isti kao build_hadamard_matrix() (features.py:107-119, provereno citanjem
koda) i train/val split koristi isti seed (SEED+500+fold_idx, kao u
MLPPairClassifier konstruktoru u loco_mlp_hadamard_1548.py) -- pa je
train_test_split() na indeksima matematicki IDENTICAN split kao da je
uradjen direktno na X (hadamard matrici), samo sto X ovde jos ne postoji
kao gotov numpy niz (racuna se live, po batch-u).

JEDINA stvarna razlika od baseline-a: embeddinzi vise nisu precomputed
(numpy iz pickle-a) vec live ESM-2 forward, sa poslednjim transformer
blokom trenabilnim -- sve ostalo bit-za-bit ista trening procedura.
ESM_LR/ESM_WEIGHT_DECAY/GRAD_CLIP_NORM ispod NEMAJU ekvivalent u baseline-u
(fine-tuning ESM-a tamo ne postoji) -- to su jedini genuinski NOVI
hiperparametri, neizbezni jer opisuju mehanizam koji baseline nema.

CENA: 44 nezavisna treninga sa live 650M forward/backward na SVAKOM od
~140+ batch-eva (64 para/batch) svake od do 300 epoha -- ovo je namerno
skupo, ne pojednostavljeno radi brzine. Ova masina NEMA GPU
(torch.cuda.is_available()==False) -- skript to detektuje i za pravi run
treba GPU/klaster; SMOKE_TEST_MAX_FOLDS ispod je SAMO za proveru da kod ne
puca (par foldova/epoha na CPU-u), nikad izvor stvarnih brojeva.

OOM BEZBEDNOST, 2026-09-12 (treca dopuna, korisnicki zahtev -- tri potvrde pre punog
44-fold run-a na klasteru):
  1) AdamW optimizer se pravi IZNOVA svaki fold (linija sa "optimizer = torch.optim.AdamW"
     je UNUTAR for fold_idx petlje) -- momentum (exp_avg/exp_avg_sq) ne prelazi izmedju foldova,
     isto kao sto svaki MLPPairClassifier.fit() poziv u baseline-u pravi nov optimizer.
  2) Validacija/early stopping je linija-po-liniju identicna mlp.py (val_auc > best_val_auc,
     best_state kloniran pri poboljsanju, no_improve += 1 inace, break kad no_improve >= patience).
  3) TRAIN_BATCH_SIZE=64 OSTAJE parametar treninga (efektivni batch, isto sto i baseline) --
     NIJE pretpostavljeno da stane u GPU memoriju. run_train_batch() ga automatski deli na
     manje "mikro-batch-eve" (gradient accumulation) SAMO ako GPU prijavi OOM, matematicki
     dokazano ekvivalentno (mean BCE + L2 skalirani razmerom mikro/pun pre backward-a, zbir
     gradijenata = gradijent punog batch-a) -- broj optimizer.step() poziva po epohi i
     efektivni batch NIKAD se ne menjaju, menja se SAMO koliko parova stoji u memoriji
     istovremeno. embed_ids_safe() radi isto za validaciju/finalni pool-embed (bez
     accumulation-a, jer tu nema backward-a). Nijedan hiperparametar (NEG_PER_POS, MAX_EPOCHS,
     PATIENCE, MAX_LENGTH, MLP arhitektura) se NIKAD ne menja zbog OOM-a -- samo mikro-batch/
     eval-chunk velicina, sto je mehanicki detalj izvrsavanja, ne trening procedura.

GOOGLE COLAB / KLASTER, 2026-09-12 dopune:
  - REPO_ROOT je sada citljivo iz env var ALERGRAF_ROOT (podrazumevano isto
    kao na dev masini) -- na Colab-u postavi npr. ALERGRAF_ROOT=/content/drive/MyDrive/ALERGRAF
    posle mount-ovanja Drive-a, umesto rucnog menjanja svake putanje.
  - RESUMABILNOST: PER_FOLD_OUTPUT i PER_QUERY_OUTPUT se sada pisu INKREMENTALNO
    (append posle SVAKOG fold-a), ne tek na kraju -- ako Colab sesija pukne na
    fold-u 30/44 (free tier ~12h limit + honestni disconnect rizik), naredno
    pokretanje ISTOG skripta (istih putanja, npr. na Drive-u) AUTOMATSKI
    preskace vec zavrsene foldove i nastavlja gde je stalo. Finalni summary se
    RACUNA IZ FAJLOVA NA DISKU (ne iz in-memory akumulatora ove sesije), pa je
    tacan i posle vise parcijalnih pokretanja.
  - FP16 (autocast) + GradScaler kad je device=cuda -- smanjuje memoriju i
    ubrzava (LoRA skript je ovo koristio iz istog razloga). Na CPU ostaje fp32
    (autocast na CPU-u ionako ne pomaze za ovaj tip racunanja).

Izlaz:
    output/partial_finetune_esm2_650m_lastblock_mlp_hadamard_1548_per_fold.csv
    output/partial_finetune_esm2_650m_lastblock_mlp_hadamard_1548_per_query.csv
    output/partial_finetune_esm2_650m_lastblock_mlp_hadamard_1548_summary.txt
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EsmModel

REPO_ROOT = Path(os.environ.get("ALERGRAF_ROOT", "/home/lana/ALERGRAF"))
sys.path.insert(0, str(REPO_ROOT))
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.evaluation import retrieval_evaluate  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.base import PairClassifier  # noqa: E402
from ml.pipeline.models.classifiers.mlp import PairMLP  # noqa: E402  -- ISTA arhitektura kao MLP(hadamard) baseline

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
N_TRAINABLE_BLOCKS = 1          # poslednji blok; staviti 2 za "poslednja 2 bloka"
CLEAN_ALLERGENS = REPO_ROOT / "output" / "clean_allergens.csv"
EMBEDDINGS_FOR_UNIVERSE = REPO_ROOT / "embeddings" / "embeddings.pkl"   # SAMO za identican candidate pool/foldove
METADATA = REPO_ROOT / "embeddings" / "embeddings.parquet"
GOLD = REPO_ROOT / "output" / "cross_reactive_1548.csv"
_OUT_PREFIX = REPO_ROOT / "output" / "partial_finetune_esm2_650m_lastblock_mlp_hadamard_1548"
PER_FOLD_OUTPUT = Path(f"{_OUT_PREFIX}_per_fold.csv")
PER_QUERY_OUTPUT = Path(f"{_OUT_PREFIX}_per_query.csv")
SUMMARY_OUTPUT = Path(f"{_OUT_PREFIX}_summary.txt")

SEED = 42
NEG_PER_POS = 10                 # IDENTICNO baseline-u (ml/loco_mlp_hadamard_1548.py)
MAX_LENGTH = 1022                 # IDENTICNO baseline-u (isti MAX_LENGTH kao embeddings/generate_embedidngs.py) -- koristi se SVUDA (trening i inference), nema posebnog "kraceg" limita za trening
TRAIN_BATCH_SIZE = 64             # IDENTICNO baseline-u (MLP_HADAMARD_PARAMS batch_size=64, parova po batch-u)
MAX_EPOCHS = 300                  # IDENTICNO baseline-u
PATIENCE = 20                     # IDENTICNO baseline-u
VAL_FRACTION = 0.15               # IDENTICNO baseline-u
MLP_LR = 1e-2                     # IDENTICNO baseline-u (MLP_HADAMARD_PARAMS learning_rate)
MLP_L2_LAMBDA = 1e-3              # IDENTICNO baseline-u (MLP_HADAMARD_PARAMS l2_lambda)
# --- Sledeca tri NEMAJU ekvivalent u baseline-u -- ESM fine-tuning tamo ne postoji, pa su
# ovo jedini genuinski novi hiperparametri (opisuju MEHANIZAM koji baseline nema, ne
# "jeftiniju verziju" postojeceg): ---
ESM_LR = 2e-5                     # bitno manji od MLP LR, kako korisnikov plan trazi
ESM_WEIGHT_DECAY = 0.01           # standardan AdamW weight decay za transformer fine-tuning
GRAD_CLIP_NORM = 1.0              # samo za ESM trenabilne parametre -- plan §7, transformer fine-tuning moze imati nestabilne gradijente

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type != "cuda":
    print("\n*** UPOZORENJE: nema GPU-a na ovoj masini. ***")
    print("Pun 44-fold trening (live 650M forward/backward svaki batch) na CPU-u")
    print("bi trajao redovima velicine duze nego sto je izvodljivo (dani, ne minuti/sati).")
    print("Ovaj skript je namenjen pokretanju na GPU (lokalna masina ili klaster).")
    print("Nastavljam samo da bih pokazao ispravnost pipeline-a na SMANJENOM broju foldova/epoha")
    print("-- videti SMOKE_TEST_MAX_FOLDS ispod. Za pun rezultat, pokrenuti na GPU.\n")

SMOKE_TEST_MAX_FOLDS = None if device.type == "cuda" else 2   # None = svi foldovi (GPU); 2 = brz smoke-test na CPU
USE_FP16 = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16)


def sample_negatives(protein_pool, n_needed, seed, positive_pair_set):
    """Identicno ml/loco_mlp_hadamard_1548.py -- ista logika/seed shema svuda u projektu."""
    local_rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)
    unlabeled = set()
    max_attempts = n_needed * 50 + 2000
    attempts = 0
    while len(unlabeled) < n_needed and attempts < max_attempts:
        a, b = local_rng.choice(pool, size=2, replace=False)
        pair = tuple(sorted((a, b)))
        attempts += 1
        if pair in positive_pair_set or pair in unlabeled:
            continue
        unlabeled.add(pair)
    return sorted(unlabeled)


# =====================================================
# LOAD CANDIDATE UNIVERSE / GOLD PAIRS / LOCO FOLDS (identicno frozen baseline-u)
# =====================================================

print("Loading dataset (frozen embeddings.pkl SAMO za universe/fold parity)...")
dataset = load_dataset(EMBEDDINGS_FOR_UNIVERSE, METADATA, GOLD)
folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS} (mora biti 44, isto kao output/loco_mlp_hadamard_1548_summary.txt)")

allergens = pd.read_csv(CLEAN_ALLERGENS)
id_to_seq_full = dict(zip(allergens["allergen_id"], allergens["fasta_sequence"]))
missing_seq = [aid for aid in dataset.all_ids if aid not in id_to_seq_full or not str(id_to_seq_full[aid]).strip()]
assert not missing_seq, f"{len(missing_seq)} proteins in candidate universe missing a FASTA sequence: {missing_seq[:5]}"
id_to_seq = {aid: id_to_seq_full[aid] for aid in dataset.all_ids}

# =====================================================
# MODEL: ESM-2 650M, ONLY last N_TRAINABLE_BLOCKS trainable
# =====================================================

print(f"\nLoading tokenizer + {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME)
model.to(device)

for p in model.parameters():
    p.requires_grad = False
trainable_blocks = model.encoder.layer[-N_TRAINABLE_BLOCKS:]
for layer in trainable_blocks:
    for p in layer.parameters():
        p.requires_grad = True

esm_trainable_params = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in esm_trainable_params)
n_total = sum(p.numel() for p in model.parameters())
print(f"ESM-2 trainable: {n_trainable:,} / {n_total:,} params ({100*n_trainable/n_total:.2f}%) "
      f"-- last {N_TRAINABLE_BLOCKS} block(s)")

# Sacuvano PRETRAINED stanje trenabilnih blokova -- reset PRE SVAKOG fold-a (plan §9,
# §6: svaki fold i svaki eksperiment kreće od ISTOG pretrained checkpoint-a, nikad
# od stanja koje je "nasledio" prethodni fold).
initial_trainable_state = [
    {name: p.detach().clone() for name, p in layer.named_parameters()} for layer in trainable_blocks
]


def reset_trainable_blocks():
    for layer, saved_state in zip(trainable_blocks, initial_trainable_state):
        with torch.no_grad():
            for name, p in layer.named_parameters():
                p.copy_(saved_state[name])


def embed_batch(seqs, max_length, no_grad):
    tokens = tokenizer(seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        if USE_FP16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**tokens).last_hidden_state
        else:
            out = model(**tokens).last_hidden_state
    out = out.float()  # nazad na fp32 pre pooling-a -- hadamard/MLP dalje rade u fp32
    mask = tokens["attention_mask"].unsqueeze(-1).expand(out.size()).float()
    summed = (out * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _is_oom(e: RuntimeError) -> bool:
    return "out of memory" in str(e).lower()


# Mutabilne "kutije" (ne obicne varijable, da funkcije ispod mogu da ih trajno smanje bez
# global/nonlocal gimnastike) -- OOM-backoff je ISKLJUCIVO mehanicki detalj (koliko se drzi
# u memoriji ISTOVREMENO), NE menja trening proceduru: efektivni batch (broj parova po
# optimizer.step() pozivu) ostaje TRAIN_BATCH_SIZE bez obzira na to koliko se ovo smanji.
_micro_batch_size = [TRAIN_BATCH_SIZE]   # za trening (gradient accumulation, videti run_train_batch)
_eval_chunk_size = [64]                   # za validaciju/finalni pool embed (obican no_grad forward, bez accumulation-a)


def embed_ids_safe(ids, max_length=MAX_LENGTH):
    """Embed-uje listu id-jeva (no_grad, inference -- validacija ili finalni pool pass), u
    komadima cija se velicina AUTOMATSKI prepolovi ako GPU prijavi OOM (i ostaje smanjena za
    ostatak run-a). Nema gradient accumulation-a ovde jer nema backward-a -- prost retry."""
    model.eval()
    out = {}
    i = 0
    while i < len(ids):
        chunk = ids[i:i + _eval_chunk_size[0]]
        try:
            seqs = [id_to_seq[aid] for aid in chunk]
            emb = embed_batch(seqs, max_length=max_length, no_grad=True).cpu().numpy()
            for aid, vec in zip(chunk, emb):
                out[aid] = vec
            i += len(chunk)
        except RuntimeError as e:
            if not _is_oom(e) or _eval_chunk_size[0] <= 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _eval_chunk_size[0] = max(1, _eval_chunk_size[0] // 2)
            print(f"  [OOM] (eval) smanjujem chunk na {_eval_chunk_size[0]}", flush=True)
    return out


def embed_pool_no_grad(ids, max_length=MAX_LENGTH):
    """Finalni embedding pass za CEO candidate pool (posle treninga tog fold-a), no_grad --
    koristi se SAMO za evaluaciju (retrieval_evaluate / cosine), nikad tokom treninga."""
    return embed_ids_safe(ids, max_length=max_length)


def run_train_batch(batch_pairs, batch_y, optimizer, mlp_head):
    """Trenira JEDAN 'pravi' batch (velicine TRAIN_BATCH_SIZE=64, isto sto i baseline
    MLP_HADAMARD_PARAMS batch_size) preko gradient accumulation-a u komadima od
    _micro_batch_size[0] parova. Efektivni batch i broj optimizer.step() poziva po epohi
    OSTAJU IDENTICNI TRAIN_BATCH_SIZE-u bez obzira na mikro-velicinu -- matematicki
    dokazano ekvivalentno jednom pravom batch-u: svaka mikro-BCE (vec mean() po mikro-batch-u)
    se skalira razmerom (mikro/pun) pre backward-a, pa zbir gradijenata svih mikro-koraka
    jednak je gradijentu srednje BCE preko celog batch-a (isto za eksplicitan L2 clan).
    Vraca raw BCE loss (bez L2 clana) za logovanje, isto sto je originalni kod logovao.
    Ako OOM udari cak i na mikro_batch=1, greska se propagira -- nema vise sta bezbedno da
    se smanji bez diranja MAX_LENGTH/arhitekture/hiperparametara, sto korisnik izricito trazi
    da se ne dira."""
    n_full = len(batch_pairs)
    while True:
        try:
            optimizer.zero_grad(set_to_none=True)
            mb = _micro_batch_size[0]
            total_raw_loss = 0.0
            for mstart in range(0, n_full, mb):
                micro_pairs = batch_pairs[mstart:mstart + mb]
                micro_y = batch_y[mstart:mstart + mb]
                frac = len(micro_pairs) / n_full

                unique_ids = sorted({pid for pair in micro_pairs for pid in pair})
                unique_seqs = [id_to_seq[pid] for pid in unique_ids]
                id_pos = {pid: i for i, pid in enumerate(unique_ids)}
                embs = embed_batch(unique_seqs, max_length=MAX_LENGTH, no_grad=False)
                idx_a = torch.tensor([id_pos[a] for a, b in micro_pairs])
                idx_b = torch.tensor([id_pos[b] for a, b in micro_pairs])
                logits = mlp_head(embs[idx_a] * embs[idx_b])

                micro_bce = criterion(logits, micro_y)  # vec mean() po mikro-batch-u
                micro_l2 = MLP_L2_LAMBDA * sum((p ** 2).sum() for p in mlp_head.parameters())
                scaler.scale((micro_bce + micro_l2) * frac).backward()
                total_raw_loss += micro_bce.item() * frac

            scaler.unscale_(optimizer)  # mora pre clip_grad_norm_ -- clip radi na stvarnim gradijentima
            torch.nn.utils.clip_grad_norm_(esm_trainable_params, GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            return total_raw_loss
        except RuntimeError as e:
            if not _is_oom(e) or _micro_batch_size[0] <= 1:
                raise
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _micro_batch_size[0] = max(1, _micro_batch_size[0] // 2)
            print(f"  [OOM] smanjujem micro-batch na {_micro_batch_size[0]} "
                  f"(efektivni batch ostaje {TRAIN_BATCH_SIZE}, gradient accumulation, "
                  f"nista drugo u trening proceduri se ne menja)", flush=True)


class FineTunedPairClassifier(PairClassifier):
    """Adapter: MLP glava vec istrenirana (na live-fine-tuned embeddinzima tog fold-a),
    embedding_matrix je POST-fine-tuning stanje candidate pool-a (fiksno u trenutku
    evaluacije -- ESM-2 se dalje ne menja). score_all koristi ISTI hadamard mehanizam
    kao MLPPairClassifier(input_encoding='hadamard', standardize=False)."""

    def __init__(self, mlp_head: nn.Module):
        super().__init__()
        self.mlp_head = mlp_head
        self.mlp_head.eval()

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Vec istreniran spolja -- vidi trening petlju ispod.")

    def score_all(self, query_id) -> np.ndarray:
        query_vec = self.embedding_matrix[self.id_to_index[query_id]]
        hadamard = self.embedding_matrix * query_vec[None, :]
        with torch.no_grad():
            logits = self.mlp_head(torch.from_numpy(hadamard.astype(np.float32)))
            return torch.sigmoid(logits).numpy()


# =====================================================
# LOCO FOLD LOOP
# =====================================================

overall_start = time.time()

n_folds_to_run = K_FOLDS if SMOKE_TEST_MAX_FOLDS is None else min(SMOKE_TEST_MAX_FOLDS, K_FOLDS)
if SMOKE_TEST_MAX_FOLDS is not None:
    print(f"\n*** SMOKE TEST: pokrecem samo {n_folds_to_run}/{K_FOLDS} foldova (nema GPU-a) ***\n")

completed_folds = set()
if PER_FOLD_OUTPUT.exists():
    completed_folds = set(pd.read_csv(PER_FOLD_OUTPUT)["fold"].tolist())
    print(f"RESUME: {len(completed_folds)} fold(ova) vec zavrseno (nadjeno u {PER_FOLD_OUTPUT}), preskacem ih.")

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds[:n_folds_to_run]):
    if fold_idx in completed_folds:
        continue
    fold_start = time.time()
    reset_trainable_blocks()

    train_ids = {pid for p in train_pairs for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = len(train_pairs) * NEG_PER_POS
    train_negatives = sample_negatives(train_ids, n_train_neg, SEED + fold_idx, dataset.positive_pair_set)

    pairs_all = [(p["id_1"], p["id_2"]) for p in train_pairs] + list(train_negatives)
    labels_all = np.array([1] * len(train_pairs) + [0] * len(train_negatives), dtype=np.float32)

    # mlp_seed = SEED+500+fold_idx -- IDENTICNO MLPPairClassifier(seed=SEED+500+fold_idx) pozivu u
    # ml/loco_mlp_hadamard_1548.py (sample_negatives ostaje na SEED+fold_idx, kao tamo). Split na
    # indeksima pairs_all (isti redosled kao build_hadamard_matrix: pozitivni pa negativni) sa istim
    # seed-om je matematicki identican split kao da je uradjen direktno na gotovoj X matrici.
    mlp_seed = SEED + 500 + fold_idx
    idx_fit, idx_val = train_test_split(
        np.arange(len(pairs_all)), test_size=VAL_FRACTION, random_state=mlp_seed, stratify=labels_all)

    torch.manual_seed(mlp_seed)  # IDENTICNO MLPPairClassifier.__init__ -- reprodukuje isto tezinsko init PairMLP-a
    mlp_head = PairMLP(input_dim=1280, hidden_dims=[32], dropout=[0.3])
    n_pos_fit = float(labels_all[idx_fit].sum())
    n_neg_fit = float((labels_all[idx_fit] == 0).sum())
    pos_weight = torch.tensor(n_neg_fit / n_pos_fit, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW([
        {"params": mlp_head.parameters(), "lr": MLP_LR, "weight_decay": 0.0},
        {"params": esm_trainable_params, "lr": ESM_LR, "weight_decay": ESM_WEIGHT_DECAY},
    ])

    batch_rng = np.random.default_rng(mlp_seed)  # IDENTICNO baseline-u (batch_rng = np.random.default_rng(self.seed))
    best_val_auc, best_mlp_state, best_esm_state, no_improve = -np.inf, None, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        mlp_head.train()
        perm = batch_rng.permutation(len(idx_fit))
        epoch_losses = []
        for start in range(0, len(perm), TRAIN_BATCH_SIZE):
            batch_pos = idx_fit[perm[start:start + TRAIN_BATCH_SIZE]]
            batch_pairs = [pairs_all[i] for i in batch_pos]
            batch_y = torch.from_numpy(labels_all[batch_pos])
            raw_loss = run_train_batch(batch_pairs, batch_y, optimizer, mlp_head)
            epoch_losses.append(raw_loss)

        model.eval()
        mlp_head.eval()
        with torch.no_grad():
            val_pairs = [pairs_all[i] for i in idx_val]
            val_y = labels_all[idx_val]
            unique_ids = sorted({pid for pair in val_pairs for pid in pair})
            emb_dict = embed_ids_safe(unique_ids, max_length=MAX_LENGTH)
            embs = torch.from_numpy(np.array([emb_dict[pid] for pid in unique_ids], dtype=np.float32))
            id_pos = {pid: i for i, pid in enumerate(unique_ids)}
            idx_a = torch.tensor([id_pos[a] for a, b in val_pairs])
            idx_b = torch.tensor([id_pos[b] for a, b in val_pairs])
            val_logits = mlp_head(embs[idx_a] * embs[idx_b])
            val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(val_y, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_mlp_state = {k: v.clone() for k, v in mlp_head.state_dict().items()}
            best_esm_state = [{n: p.detach().clone() for n, p in layer.named_parameters()}
                               for layer in trainable_blocks]
            no_improve = 0
        else:
            no_improve += 1
        print(f"  fold {fold_idx+1}/{n_folds_to_run} epoch {epoch}/{MAX_EPOCHS} "
              f"train_loss={np.mean(epoch_losses):.4f} val_auc={val_auc:.4f} "
              f"(best={best_val_auc:.4f}, no_improve={no_improve})", flush=True)
        if no_improve >= PATIENCE:
            break

    mlp_head.load_state_dict(best_mlp_state)
    for layer, saved_state in zip(trainable_blocks, best_esm_state):
        with torch.no_grad():
            for name, p in layer.named_parameters():
                p.copy_(saved_state[name])
    mlp_head.eval()
    model.eval()

    fold_embeddings = embed_pool_no_grad(dataset.all_ids)
    fold_embedding_matrix = np.array([fold_embeddings[aid] for aid in dataset.all_ids], dtype=np.float64)
    fold_cosine_matrix = cosine_similarity(fold_embedding_matrix)

    classifier = FineTunedPairClassifier(mlp_head)
    classifier.set_pool(fold_embedding_matrix, dataset.id_to_index)
    fold_df = retrieval_evaluate(test_pairs, classifier, fold_embedding_matrix, dataset.id_to_index,
                                   cosine_matrix=fold_cosine_matrix)
    fold_df["fold"] = fold_idx

    fold_row = {
        "fold": fold_idx, "component_size": len(test_ids), "n_queries": len(fold_df),
        "best_val_auc": best_val_auc, "stopped_epoch": epoch,
        "cosine_mrr": fold_df["cosine_reciprocal_rank"].mean(),
        "mlp_hadamard_finetuned_mrr": fold_df["model_reciprocal_rank"].mean(),
    }

    # Upis ODMAH (ne na kraju svih foldova) -- ako sesija pukne posle ovog fold-a,
    # naredno pokretanje ce ga videti u completed_folds i preskociti.
    pd.DataFrame([fold_row]).to_csv(PER_FOLD_OUTPUT, mode="a", header=not PER_FOLD_OUTPUT.exists(), index=False)
    fold_df.to_csv(PER_QUERY_OUTPUT, mode="a", header=not PER_QUERY_OUTPUT.exists(), index=False)

    elapsed_fold = time.time() - fold_start
    elapsed_total = time.time() - overall_start
    print(f"fold {fold_idx+1}/{n_folds_to_run} DONE (size={len(test_ids)}, queries={len(fold_df)}) -- "
          f"cosine={fold_row['cosine_mrr']:.4f} "
          f"mlp_finetuned={fold_row['mlp_hadamard_finetuned_mrr']:.4f} "
          f"({elapsed_fold/60:.1f} min this fold, {elapsed_total/60:.1f} min total)\n", flush=True)

reset_trainable_blocks()  # ostavi model u pretrained stanju na kraju skripta

# =====================================================
# SUMMARY -- iz fajlova na disku (ne iz in-memory akumulatora ove sesije), da bude
# tacan i kad je run resumovan preko vise (Colab) sesija.
# =====================================================

per_fold_df = pd.read_csv(PER_FOLD_OUTPUT)
all_df = pd.read_csv(PER_QUERY_OUTPUT)

cos_macro = per_fold_df["cosine_mrr"].to_numpy()
mlp_macro = per_fold_df["mlp_hadamard_finetuned_mrr"].to_numpy()
delta = mlp_macro - cos_macro
se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else float("nan")

if len(per_fold_df) < K_FOLDS:
    print(f"\n*** DELIMICAN REZULTAT: {len(per_fold_df)}/{K_FOLDS} foldova zavrseno. "
          f"Ponovo pokreni skript da nastavis. ***\n")

summary_lines = [
    "=" * 70,
    f"Partial fine-tuning ESM-2 650M (last {N_TRAINABLE_BLOCKS} block) + MLP(hadamard), "
    f"LOCO ({len(per_fold_df)}/{K_FOLDS} folds run)"
    + (" -- DELIMICAN REZULTAT" if len(per_fold_df) < K_FOLDS else ""),
    "=" * 70,
    f"Device: {device}" + (" -- SMOKE TEST, not full result" if SMOKE_TEST_MAX_FOLDS else ""),
    f"Total runtime: {(time.time()-overall_start)/60:.1f} min", "",
    "MACRO (unweighted mean across component-folds):",
    f"  cosine (fine-tuned emb)      MRR: {cos_macro.mean():.4f} +/- {cos_macro.std(ddof=1):.4f}",
    f"  MLP(hadamard) fine-tuned     MRR: {mlp_macro.mean():.4f} +/- {mlp_macro.std(ddof=1):.4f}", "",
    "MICRO (query-weighted, pooled -- najpouzdaniji broj):",
    f"  cosine (fine-tuned emb)      MRR: {all_df['cosine_reciprocal_rank'].mean():.4f}",
    f"  MLP(hadamard) fine-tuned     MRR: {all_df['model_reciprocal_rank'].mean():.4f}", "",
    f"Paired delta MLP(hadamard) vs cosine (this run): {delta.mean():+.4f} (SE {se:.4f})", "",
    "--- Referentni frozen 650M baseline (output/loco_mlp_hadamard_1548_summary.txt, 44 folds) ---",
    "  frozen 650M   MLP(hadamard) MICRO MRR: 0.1209, MACRO MRR: 0.3282",
    "  (poredjenje sa gornjim brojevima je INFORMATIVNO ako je len(per_fold_df) < 44 -- "
    "za pravo poredjenje treba svih 44 folda, na GPU-u)",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
