"""
Izvlaci per-residue ESMFold trunk "single" reprezentaciju (s_s), PROJEKTOVANU
na 384-dim TACNO onako kako ulazi u Structure Module -- NE finalne 3D
koordinate/frames/pLDDT/PAE/distograme/pair reprezentaciju.

===========================================================================
VAZNA TEHNICKA NAPOMENA -- verifikovano DIREKTNO iz izvornog koda
transformers/models/esm/modeling_esmfold.py (ne pretpostavka):
===========================================================================
EsmForProteinFolding.forward() vraca output.s_s, ALI to je SIROVO 1024-dim
trunk stanje (config.esmfold_config.trunk.sequence_state_dim=1024) -- NE
384-dim reprezentacija koja stvarno ulazi u Structure Module. Ta 384-dim
verzija se racuna INTERNO preko self.trunk.trunk2sm_s (nn.Linear(1024,384),
videti modeling_esmfold.py liniju ~1857/1914) i NIKAD se ne cuva u izlaznom
objektu -- prosledjuje se direktno u Structure Module i gubi. Ovaj skript
zato RUCNO primenjuje ISTU projekciju (model.trunk.trunk2sm_s) na output.s_s
POSLE forward-a, dajuci matematicki IDENTICAN tenzor onome koji je Structure
Module stvarno video (deterministicka linearna transformacija naucenim,
ucitanim tezinama -- ne aproksimacija).

Ovo NIJE OpenFold (kao ranija verzija ovog eksperimenta,
hpclab/generate_alphafold_trunk_embeddings.py, single-sequence rezim bez
prave MSA pretrage) -- ESMFold je STVARNO dizajniran da radi BEZ MSA
pretrage (koristi ESM-2 3B jezicki model umesto Evoformer-a nad MSA), pa je
ovde "single-sequence" nacin rada ORIGINALNI, ne degradiran workaround.

===========================================================================
GPU UPOZORENJE -- na osnovu VEC POTVRDJENOG OOM-a na istom klasteru
===========================================================================
ESMFold INTERNO ukljucuje CEO ESM-2 3B jezicki model (esm_type="esm2_3B" u
config-u) PLUS folding trunk (48 blokova, sequence_state_dim=1024) PLUS
structure module -- ukupno VECI model od samog ESM-2 3B, koji je vec OOM-ovao
na GPU sa 3.94GB VRAM (facebook/esm2_t36_3B_UR50D pokusaj, ranije ove sesije).
Realno ocekivanje: treba GPU sa BAREM 16GB VRAM (FP16) da ovo uopste stane.
Skript proverava dostupan VRAM na startu i glasno upozorava, ali NE odustaje
automatski -- probaj napravi_procenu(), pa odluci sama da li da nastavis.

Ulaz:
    clean_allergens.csv (kolone: allergen_id, fasta_sequence)

Izlaz:
    residue_embeddings_af_trunk.pkl
    {allergen_id: np.ndarray(shape=(L,384), dtype=float32)}

Pokretanje:
    python3 generate_af_trunk_embeddings.py
"""

from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

# ======================================================
# Configuration -- relativne putanje, pokreni iz foldera gde je
# clean_allergens.csv transferovan
# ======================================================

INPUT_CSV = Path("clean_allergens.csv")
OUTPUT_PICKLE = Path("residue_embeddings_af_trunk.pkl")
MAX_LENGTH = 1022  # isti limit kao ostale ESM-2 skripte u ovom projektu
PRINT_EVERY = 10


def load_existing(path: Path) -> dict:
    if path.exists():
        with open(path, "rb") as f:
            existing = pickle.load(f)
        print(f"Nastavljam prethodni run: {len(existing)} proteina vec zavrseno u {path}", flush=True)
        return existing
    return {}


def save_atomic(path: Path, d: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==============================")
    print("DEVICE")
    print("==============================")
    print(device)
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f} GB")
        if vram_gb < 14:
            print(f"UPOZORENJE: ESMFold (uklj. ceo ESM-2 3B + trunk + structure module) "
                  f"vrlo verovatno NECE stati u {vram_gb:.1f}GB VRAM -- ESM-2 3B SAM je "
                  f"vec OOM-ovao na 3.94GB GPU-u ranije ove sesije, a ESMFold je veci od toga. "
                  f"Probaj svejedno (skript ce jasno prijaviti OOM ako se desi), ali ne budi "
                  f"iznenadjena ako odmah padne pri ucitavanju modela.", flush=True)
    else:
        print("UPOZORENJE: nema GPU-a -- ESMFold na CPU-u ce biti EKSTREMNO spor "
              "(sati po proteinu, ne po batch-u), prakticno neupotrebljivo bez GPU-a.")
    print()

    print(f"Loading tokenizer (facebook/esmfold_v1)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    print(f"Loading ESMFold model (facebook/esmfold_v1) -- veliki download (~15GB) pri prvom pokretanju...",
          flush=True)
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()
    # Isti razlog kao ostale ESM skripte u projektu -- ESM (BERT-stil trunk) FP16 stabilan na GPU-u,
    # dodatno OVDE bitno smanjuje VRAM (esmfold_v1 je veliki model).
    if device.type == "cuda":
        model = model.half()
        print("Using FP16 (smanjuje VRAM, ESM arhitektura je numericki stabilna u FP16)")
    print()

    df = pd.read_csv(INPUT_CSV)
    df = df[df["fasta_sequence"].notna()]
    df = df[df["fasta_sequence"] != ""]
    df = df.reset_index(drop=True)
    print("==============================")
    print("DATASET")
    print("==============================")
    print(f"Loaded {len(df)} allergens")
    print()

    embeddings = load_existing(OUTPUT_PICKLE)
    todo = [(row.allergen_id, row.fasta_sequence) for row in df.itertuples(index=False)
            if row.allergen_id not in embeddings]
    print(f"Preostalo za obradu: {len(todo)}/{len(df)}", flush=True)

    n_done, n_failed = 0, 0
    overall_start = time.time()

    with torch.no_grad():
        for i, (allergen_id, sequence) in enumerate(todo, 1):
            sequence = sequence[:MAX_LENGTH]
            try:
                inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                # output.s_s = SIROVO 1024-dim trunk stanje (videti napomena na vrhu fajla) --
                # rucno primeni ISTU projekciju koju model interno koristi pre Structure Module-a.
                trunk_s = outputs.s_s  # (1, L, 1024)
                projected = model.trunk.trunk2sm_s(trunk_s)  # (1, L, 384) -- tacan ulaz u Structure Module

                arr = projected.squeeze(0).float().cpu().numpy().astype(np.float32)
                assert arr.shape == (len(sequence), 384), \
                    f"{allergen_id}: ocekivan oblik ({len(sequence)}, 384), dobijeno {arr.shape}"
                embeddings[allergen_id] = arr
                n_done += 1

            except torch.cuda.OutOfMemoryError as e:
                n_failed += 1
                print(f"  [{i}/{len(todo)}] {allergen_id} (len={len(sequence)}): OOM, PRESKACEM: {e}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                n_failed += 1
                print(f"  [{i}/{len(todo)}] {allergen_id}: GRESKA, PRESKACEM: {e}", flush=True)
                continue

            save_atomic(OUTPUT_PICKLE, embeddings)  # posle SVAKOG proteina -- SLURM job moze biti ubijen bilo kad

            if i % PRINT_EVERY == 0 or i == len(todo):
                elapsed = time.time() - overall_start
                print(f"  [{i}/{len(todo)}] {allergen_id} (len={len(sequence)}) -> {arr.shape} "
                      f"({elapsed/60:.1f} min ukupno, {elapsed/i:.1f}s/protein prosek)", flush=True)

    total_elapsed = time.time() - overall_start
    print()
    print("==============================")
    print("DONE")
    print("==============================")
    print(f"Ukupno u {OUTPUT_PICKLE}: {len(embeddings)} proteina")
    print(f"Ovaj run: {n_done} novo obradjeno, {n_failed} greska/OOM, trajanje {total_elapsed/60:.1f} min")
    if n_failed:
        print(f"UPOZORENJE: {n_failed} proteina NIJE uspesno obradjeno -- ponovo pokreni istu komandu "
              f"(resume ce preskociti vec zavrsene, pokusace ponovo neuspele)")
    print(f"Saved: {OUTPUT_PICKLE}")


if __name__ == "__main__":
    main()
