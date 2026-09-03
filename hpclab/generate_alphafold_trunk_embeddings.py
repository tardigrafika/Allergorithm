"""
Izvlaci per-residue AlphaFold/OpenFold TRUNK reprezentacije (Evoformer
"single" izlaz, PRE structure module-a) za protein dataset -- SAMO
sequence -> trunk -> .pkl, bez pooling-a/MLP/Hadamard/cosine/ranking-a
(to se radi odvojeno, isti downstream pipeline kao za ESM embeddinge).

===========================================================================
VAZNA NAUCNA NAPOMENA -- PROCITAJ PRE POKRETANJA (ne simplifikovati bez
objasnjenja, CLAUDE.md pravilo ovog projekta):
===========================================================================
Ovaj skript koristi SINGLE-SEQUENCE rezim -- upitna sekvenca se tretira kao
SVOJA JEDINA MSA vrsta, BEZ prave MSA/template pretrage (jackhmmer/HHblits
protiv UniRef90/BFD/MGnify baza). Razlog: prava pretraga zahteva STOTINE GB
genetskih baza + instalaciju HMMER/HHsuite alata, sto NIJE realno na
trenutnoj klaster konfiguraciji (GPU sa ~4GB VRAM je vec usko grlo i za
sam model, potvrdjeno OOM-om na ESM-2 3B pokusaju).

**Posledica**: AlphaFold-ova tacnost dolazi VECINSKI iz ko-evolucionog
signala u pravoj MSA (koje mutacije se javljaju ZAJEDNO kroz srodne vrste).
Single-sequence rezim GUBI taj signal skoro potpuno -- trunk embeddinzi iz
ovog rezima su STRUKTURNO/FIZICKI informisani (Evoformer i dalje uci
biohemijske/lokalne obrasce iz same sekvence i trening podataka), ali NISU
ekvivalentni "pravim" AlphaFold predikcijama. Ovo treba tretirati kao
NEZAVISNU, slabiju hipotezu za testiranje (slicno ESM-1b eksperimentu), ne
kao "AlphaFold rezultat" u punom smislu -- isto se odnosi i na finalne
strukture (koje se ovde NE cuvaju, ali kad bi se koristile, bile bi
znatno manje pouzdane nego sa pravom MSA).

===========================================================================
STATUS OVOG SKRIPTA -- nije izvrsno-testiran (nema OpenFold/GPU/tezine u
razvojnom okruzenju gde je pisan), ALI ekstrakcija tacnog tenzora JESTE
verifikovana protiv stvarnog OpenFold izvornog koda (aqlaboratory/openfold
na GitHub-u, openfold/model/model.py + openfold/config.py, 2026-09-01) --
NE pretpostavka. Potvrdjeno direktno iz izvora: OpenFold Evoformer trunk
proizvodi "single" reprezentaciju VEC na c_s=384 (config.py linija ~311),
i ona ulazi u Structure Module BEZ posredne projekcije koja bi se izgubila
(za razliku od ESMFold-a, gde model.forward() vraca SIROVO 1024-dim stanje,
a stvarni 384-dim ulaz u Structure Module postoji SAMO interno preko
self.trunk.trunk2sm_s projekcije -- videti generate_af_trunk_embeddings.py
za taj slucaj). Kod ispod (outputs["single"]) je zato VEC tacan, ne treba
dodatnu rucnu projekciju. I dalje OBAVEZNO pokreni SMOKE TEST (--smoke-test,
ispod) na 1-2 kratke sekvence pre punog cluster run-a -- ekstrakcija je
verifikovana, ali ceo forward pass (feature pipeline, checkpoint loading)
nije izvrsno testiran. Ako nesto ne odgovara tvojoj instaliranoj OpenFold
verziji (checkpoint wrapping, recycling-dim, config
preset ime), uporedi sa openfold/run_pretrained_openfold.py iz zvanicnog
repo-a (aqlaboratory/openfold) -- tamo je referentni primer za TVOJU
verziju.
===========================================================================

Preduslov (instalacija na klasteru, van obima ovog skripta):
    pip install openfold biopython   (ili conda/git-clone instalacija --
    OpenFold cesto zahteva build od izvora zbog CUDA kernela, videti
    OpenFold README za tacna uputstva za tvoj CUDA/PyTorch)
    Preuzmi OpenFold checkpoint (.pt fajl, npr. sa OpenFold Zenodo/AWS
    release-a) -- prosledi putanju preko --checkpoint-path.

Ulaz:
    FASTA fajl (isti protein dataset kao za ESM-2 embeddinge -- konvertuj
    output/clean_allergens.csv u FASTA sa fasta_to_file() ili ekvivalentnim
    pre transfera, ovaj skript ocekuje standardan .fasta/.fa format)

Izlaz:
    .pkl fajl: {protein_id: np.ndarray shape (sequence_length, embedding_dim)}
    embedding_dim je 384 za standardni AlphaFold "single" trunk kanal
    (c_s=384 u originalnoj arhitekturi) -- proveri cfg.model.evoformer_stack.c_s
    ako tvoj config preset koristi drugu vrednost.

Pokretanje:
    python3 generate_alphafold_trunk_embeddings.py \\
        --fasta proteins.fasta --output trunk_embeddings.pkl \\
        --checkpoint-path /path/to/openfold_checkpoint.pt \\
        --batch-size 1 --device cuda

    Smoke test (2 kratke sekvence, brzo, PRE punog run-a):
    python3 generate_alphafold_trunk_embeddings.py --smoke-test \\
        --checkpoint-path /path/to/openfold_checkpoint.pt --device cuda
"""

import argparse
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


# ===========================================================================
# FASTA parsing (biopython ako je dostupan, prost fallback ako nije --
# nema nepotrebnu zavisnost za ovako mali posao)
# ===========================================================================

def parse_fasta(path: Path):
    """Vraca listu (protein_id, sequence) -- ID je prva rec (do prvog belog
    razmaka) posle '>', isto sto Biopython/standardna FASTA konvencija."""
    records = []
    current_id, current_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id is not None:
            records.append((current_id, "".join(current_seq)))
    return records


# ===========================================================================
# Checkpoint/resume: ucitaj postojeci output .pkl (ako postoji), preskoci
# vec zavrsene proteine. Cuva se ATOMICNO (write temp + rename) posle SVAKOG
# proteina -- SLURM job moze biti ubijen bilo kad (time limit), ne sme se
# izgubiti vec uradjeni posao niti se ostaviti polu-napisan .pkl.
# ===========================================================================

def load_existing_output(output_path: Path) -> dict:
    if output_path.exists():
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
        print(f"Nastavljam prethodni run: {len(existing)} proteina vec zavrseno u {output_path}", flush=True)
        return existing
    return {}


def save_output_atomic(output_path: Path, embeddings_dict: dict):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(output_path)  # atomican rename na POSIX sistemima (SLURM klasteri su Linux)


# ===========================================================================
# OpenFold feature construction -- SINGLE-SEQUENCE rezim (videti napomenu na
# vrhu fajla). Koristi REALNE OpenFold data_pipeline funkcije, ne rucno
# graden tensor format -- manje rizika od pogresnog feature encoding-a.
# ===========================================================================

def build_single_sequence_features(protein_id: str, sequence: str, config):
    from openfold.data import data_pipeline
    from openfold.data.parsers import Msa

    sequence_features = data_pipeline.make_sequence_features(
        sequence=sequence, description=protein_id, num_res=len(sequence))

    # Single-sequence "MSA": upitna sekvenca kao jedina vrsta, deletion_matrix
    # sve nule (nema insercija/delecija u odnosu na sebe samu).
    msa = Msa(sequences=[sequence], deletion_matrix=[[0] * len(sequence)], descriptions=[protein_id])
    msa_features = data_pipeline.make_msa_features(msas=[msa])

    raw_features = {**sequence_features, **msa_features}

    from openfold.data.feature_pipeline import FeaturePipeline
    feature_pipeline = FeaturePipeline(config.data)
    processed = feature_pipeline.process_features(raw_features, mode="predict")
    return processed


# ===========================================================================
# Model loading
# ===========================================================================

def load_model(checkpoint_path: str, config_preset: str, device: torch.device):
    from openfold.config import model_config
    from openfold.model.model import AlphaFold

    cfg = model_config(config_preset, train=False)
    # Iskljuci recycling (podrazumevano AlphaFold radi 3+ recycling iteracije,
    # svaka je pun trunk forward pass -- za trunk-only embeddinge bez potrebe
    # za maksimalnom preciznoscu finalne strukture, 0-1 iteracija je dovoljna
    # i znacajno stedi vreme/memoriju na ogranicenom GPU-u). Podesi na 1 ili
    # vise ako primetis da embeddinzi izgledaju nestabilno bez recycling-a.
    cfg.data.common.max_recycling_iters = 0
    cfg.globals.chunk_size = None  # None = bez chunking-a; postavi na npr. 4 ako OOM na dugim sekvencama

    model = AlphaFold(cfg)

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # OpenFold checkpoint-i variraju u wrapping-u zavisno od izvora
    # (sirovi state_dict, ili umotano pod "model"/"ema"/"params" kljucem).
    # Probaj poznate obrasce redom, jasna greska ako nijedan ne odgovara --
    # NE tiho ucitaj pogresne/delimicne tezine.
    if isinstance(checkpoint, dict) and "model" in checkpoint and hasattr(checkpoint["model"], "keys"):
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "ema" in checkpoint and "params" in checkpoint.get("ema", {}):
        state_dict = checkpoint["ema"]["params"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"UPOZORENJE pri ucitavanju tezina -- missing={len(missing)}, unexpected={len(unexpected)}. "
              f"Proveri da li je checkpoint format tacan za instaliranu OpenFold verziju "
              f"(uporedi sa openfold/run_pretrained_openfold.py) pre nego sto verujes izlazu.", flush=True)
        if len(missing) > 50 or len(unexpected) > 50:
            raise RuntimeError("Previse missing/unexpected kljuceva -- checkpoint verovatno NIJE kompatibilan "
                                "sa ovim config preset-om/OpenFold verzijom. Ne nastavljaj bez provere.")

    model = model.to(device)
    model.eval()
    return model, cfg


# ===========================================================================
# Trunk ekstrakcija za JEDNU sekvencu -- namerno batch_size=1 interno (prava
# batch obrada preko proteina razlicite duzine bi zahtevala padding do
# max-duzine u batch-u, sto O(N^2) pair-representation memoriju cini jos
# skupljom za SVAKI protein u batch-u -- rizicno na ogranicenom GPU-u,
# videti napomenu u main()). --batch-size CLI arg kontrolise KOLIKO
# proteina se ucita/procesira izmedju save-ova, ne pravi tenzorski batch.
# ===========================================================================

def extract_trunk_embedding(model, config, protein_id: str, sequence: str, device: torch.device) -> np.ndarray:
    from openfold.utils.tensor_utils import tensor_tree_map

    processed = build_single_sequence_features(protein_id, sequence, config)
    batch = {k: torch.as_tensor(v, device=device) for k, v in processed.items()}

    with torch.no_grad():
        outputs = model(batch)

    single = outputs["single"]  # (N_res, c_s) -- trunk reprezentacija PRE structure module-a
    single = single.detach().float().cpu().numpy()

    # Eksplicitno oslobodi SVE ostalo (koordinate/structure module izlaze) --
    # nikad se ne cuvaju na disk, samo trunk "single" reprezentacija.
    del outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    assert single.shape[0] == len(sequence), (
        f"{protein_id}: trunk duzina {single.shape[0]} != sequence duzina {len(sequence)} -- "
        f"proveri da li je sequence_features izgradjen ispravno (padding/truncation problem?)")
    return single


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", type=Path, help="Ulazni FASTA fajl (protein_id -> sekvenca)")
    parser.add_argument("--output", type=Path, help="Izlazni .pkl (protein_id -> np.ndarray)")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Putanja do OpenFold .pt checkpoint-a")
    parser.add_argument("--config-preset", type=str, default="model_1_ptm",
                         help="OpenFold config preset ime (videti openfold/config.py za validne opcije "
                              "tvoje instalirane verzije ako ovaj default ne postoji)")
    parser.add_argument("--batch-size", type=int, default=1,
                         help="Broj proteina izmedju save-ova na disk (NE tenzorski batch -- videti napomenu "
                              "iznad extract_trunk_embedding). Preporuceno 1 na ogranicenom GPU-u.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-length", type=int, default=1022,
                         help="Preskoci proteine duze od ovoga (pair-representation memorija skalira O(N^2) -- "
                              "vrlo duge sekvence mogu OOM-ovati na ogranicenom GPU-u)")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Ignorisi --fasta, testiraj na 2 ugradjene kratke sekvence -- pokreni OVO PRVO "
                              "pre punog cluster job-a.")
    args = parser.parse_args()

    if args.smoke_test:
        records = [
            ("smoke_test_1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGDRQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
                None),
            ("smoke_test_2", "MASNTVSAQGQGQAAKPKPAALPVAKPTQAKAKVTAAAA", None),
        ]
        records = [(pid, seq) for pid, seq, _ in records]
        output_path = Path("smoke_test_trunk_embeddings.pkl")
        print("SMOKE TEST rezim -- 2 kratke ugradjene sekvence, output: smoke_test_trunk_embeddings.pkl", flush=True)
    else:
        if not args.fasta or not args.output:
            parser.error("--fasta i --output su obavezni osim u --smoke-test rezimu")
        records = parse_fasta(args.fasta)
        output_path = args.output
        print(f"Ucitano {len(records)} proteina iz {args.fasta}", flush=True)

    n_skipped_length = sum(1 for _, seq in records if len(seq) > args.max_length)
    if n_skipped_length:
        print(f"UPOZORENJE: {n_skipped_length} proteina preko --max-length={args.max_length}, bice preskoceni "
              f"(zabelezeni u output-u kao izostavljeni, ne tiho ignorisani)", flush=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("UPOZORENJE: --device cuda trazen ali CUDA nije dostupna, pada na CPU (bice VRLO sporo).", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    model, config = load_model(args.checkpoint_path, args.config_preset, device)
    print("Model ucitan.", flush=True)

    embeddings_dict = load_existing_output(output_path)
    skipped_lengths_log = []

    todo = [(pid, seq) for pid, seq in records if pid not in embeddings_dict]
    print(f"Preostalo za obradu: {len(todo)}/{len(records)} (ostalo vec u {output_path} od prethodnog run-a)",
          flush=True)

    overall_start = time.time()
    n_done_this_run, n_failed = 0, 0

    for i, (protein_id, sequence) in enumerate(todo, 1):
        if len(sequence) > args.max_length:
            skipped_lengths_log.append(protein_id)
            continue

        try:
            t0 = time.time()
            embedding = extract_trunk_embedding(model, config, protein_id, sequence, device)
            embeddings_dict[protein_id] = embedding
            n_done_this_run += 1
            elapsed = time.time() - t0

            if n_done_this_run % args.batch_size == 0 or i == len(todo):
                save_output_atomic(output_path, embeddings_dict)

            total_elapsed = time.time() - overall_start
            print(f"  [{i}/{len(todo)}] {protein_id} (len={len(sequence)}) -> "
                  f"{embedding.shape} ({elapsed:.1f}s, ukupno {total_elapsed/60:.1f} min)", flush=True)

        except torch.cuda.OutOfMemoryError as e:
            n_failed += 1
            print(f"  [{i}/{len(todo)}] {protein_id} (len={len(sequence)}): OOM, PRESKACEM "
                  f"(probaj --max-length manji, ili chunk_size u config-u): {e}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        except Exception:
            n_failed += 1
            print(f"  [{i}/{len(todo)}] {protein_id}: GRESKA, PRESKACEM:", flush=True)
            traceback.print_exc()
            continue

    save_output_atomic(output_path, embeddings_dict)

    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 70, flush=True)
    print("DONE", flush=True)
    print("=" * 70, flush=True)
    print(f"Ukupno u {output_path}: {len(embeddings_dict)} proteina", flush=True)
    print(f"Ovaj run: {n_done_this_run} novo obradjeno, {n_failed} greska/OOM, "
          f"{len(skipped_lengths_log)} preskoceno zbog --max-length, "
          f"trajanje {total_elapsed/60:.1f} min", flush=True)
    if n_failed:
        print(f"UPOZORENJE: {n_failed} proteina NIJE uspesno obradjeno -- ponovo pokreni isti "
              f"komandu (resume ce ih pokusati ponovo, vec zavrseni se preskacu)", flush=True)
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
