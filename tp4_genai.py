import argparse
import random
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


DATA_URL = "https://download.pytorch.org/tutorial/data.zip"
DATA_DIR = Path("data_tp4")
DEFAULT_CORPUS_NAME = "eng-fra.txt"

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

SEED = 42
TRAIN_RATIO = 0.9
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
MIN_FREQ = 1
EMBED_DIM = 128
HIDDEN_DIM = 256
ATTN_DIM = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 4
DEFAULT_MAX_LENGTH = 15
DEFAULT_MAX_PAIRS = 12000
DEFAULT_TEACHER_FORCING = 0.5


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower().strip())
    text = re.sub(r"([?.!,;:])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


class Vocabulary:
    def __init__(self, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}

    def build(self, token_sequences: list[list[str]]) -> None:
        counter = Counter(token for seq in token_sequences for token in seq)
        frequent_tokens = [
            token
            for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            if freq >= self.min_freq
        ]
        self.itos = SPECIAL_TOKENS + frequent_tokens
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def encode(
        self,
        tokens: list[str],
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = [self.stoi.get(token, self.unk_idx) for token in tokens]
        if add_sos:
            ids = [self.sos_idx] + ids
        if add_eos:
            ids = ids + [self.eos_idx]
        return ids

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        words = []
        for idx in token_ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if token == EOS_TOKEN:
                break
            if skip_special and token in SPECIAL_TOKENS:
                continue
            words.append(token)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.stoi[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]


def download_and_extract_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(data_dir.rglob(DEFAULT_CORPUS_NAME))
    if existing_files:
        return existing_files[0]

    zip_path = data_dir / "data.zip"
    print(f"Telechargement du corpus depuis {DATA_URL} ...")
    try:
        urllib.request.urlretrieve(DATA_URL, zip_path)
    except Exception as exc:
        raise RuntimeError(
            "Impossible de telecharger le corpus. "
            "Telecharge manuellement https://download.pytorch.org/tutorial/data.zip "
            "puis place le fichier eng-fra.txt dans data_tp4/."
        ) from exc

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)

    extracted_files = list(data_dir.rglob(DEFAULT_CORPUS_NAME))
    if not extracted_files:
        raise FileNotFoundError("eng-fra.txt introuvable apres extraction de data.zip.")
    return extracted_files[0]


def load_parallel_corpus(
    corpus_path: Path,
    max_pairs: int | None,
    max_length: int,
) -> list[tuple[list[str], list[str]]]:
    pairs: list[tuple[list[str], list[str]]] = []

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            src_tokens = tokenize(parts[0])
            tgt_tokens = tokenize(parts[1])

            if not src_tokens or not tgt_tokens:
                continue
            if len(src_tokens) > max_length or len(tgt_tokens) > max_length:
                continue

            pairs.append((src_tokens, tgt_tokens))
            if max_pairs is not None and len(pairs) >= max_pairs:
                break

    if not pairs:
        raise ValueError("Aucune paire de phrases valide n'a ete trouvee.")
    return pairs


def build_toy_pairs() -> list[tuple[list[str], list[str]]]:
    raw_pairs = [
        ("hello .", "bonjour ."),
        ("i am cold .", "j'ai froid ."),
        ("i am tired .", "je suis fatigue ."),
        ("he is calm .", "il est calme ."),
        ("she is kind .", "elle est gentille ."),
        ("we are ready .", "nous sommes prets ."),
        ("they are here .", "ils sont ici ."),
        ("open the door .", "ouvre la porte ."),
        ("close the window .", "ferme la fenetre ."),
        ("thank you .", "merci ."),
        ("good night .", "bonne nuit ."),
        ("see you soon .", "a bientot ."),
    ]
    return [(tokenize(src), tokenize(tgt)) for src, tgt in raw_pairs]


def split_pairs(
    pairs: list[tuple[list[str], list[str]]],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> tuple[list[tuple[list[str], list[str]]], list[tuple[list[str], list[str]]]]:
    shuffled_pairs = pairs[:]
    random.Random(seed).shuffle(shuffled_pairs)
    split_idx = max(1, int(len(shuffled_pairs) * train_ratio))
    split_idx = min(split_idx, len(shuffled_pairs) - 1)
    return shuffled_pairs[:split_idx], shuffled_pairs[split_idx:]


class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[list[str], list[str]]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
    ) -> None:
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_tokens, tgt_tokens = self.pairs[idx]
        src_ids = self.src_vocab.encode(src_tokens, add_eos=True)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens, add_sos=True, add_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def make_collate_fn(src_pad_idx: int, tgt_pad_idx: int):
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)

        src_lengths = torch.tensor([seq.size(0) for seq in src_batch], dtype=torch.long)
        tgt_lengths = torch.tensor([seq.size(0) for seq in tgt_batch], dtype=torch.long)

        max_src_len = int(src_lengths.max().item())
        max_tgt_len = int(tgt_lengths.max().item())

        padded_src = torch.stack(
            [F.pad(seq, (0, max_src_len - seq.size(0)), value=src_pad_idx) for seq in src_batch]
        )
        padded_tgt = torch.stack(
            [F.pad(seq, (0, max_tgt_len - seq.size(0)), value=tgt_pad_idx) for seq in tgt_batch]
        )

        return {
            "src": padded_src,
            "src_lengths": src_lengths,
            "tgt": padded_tgt,
            "tgt_lengths": tgt_lengths,
        }

    return collate_fn


@dataclass
class ModelConfig:
    name: str
    embedding_dim: int = EMBED_DIM
    encoder_hidden_dim: int = HIDDEN_DIM
    decoder_hidden_dim: int = HIDDEN_DIM
    attention_dim: int = ATTN_DIM
    encoder_layers: int = 1
    decoder_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.2


class AdditiveAttention(nn.Module):
    def __init__(self, decoder_hidden_dim: int, encoder_output_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(encoder_output_dim, attention_dim, bias=False)
        self.score_proj = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_proj(decoder_hidden).unsqueeze(1)
        keys = self.key_proj(encoder_outputs)
        scores = self.score_proj(torch.tanh(query + keys)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        decoder_hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_dim * self.num_directions
        self.hidden_bridge = nn.Linear(self.output_dim, decoder_hidden_dim)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1),
        )

        batch_size = src.size(0)
        hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_dim)
        if self.num_directions == 2:
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        else:
            hidden = hidden[:, 0]
        hidden = torch.tanh(self.hidden_bridge(hidden))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        decoder_hidden_dim: int,
        encoder_output_dim: int,
        attention_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(
            decoder_hidden_dim=decoder_hidden_dim,
            encoder_output_dim=encoder_output_dim,
            attention_dim=attention_dim,
        )

        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            input_dim = embedding_dim + encoder_output_dim if layer_idx == 0 else decoder_hidden_dim
            self.cells.append(nn.GRUCell(input_dim, decoder_hidden_dim))

        self.output_layer = nn.Linear(
            decoder_hidden_dim + encoder_output_dim + embedding_dim,
            vocab_size,
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_tokens))
        context, attention_weights = self.attention(
            decoder_hidden=hidden_states[-1],
            encoder_outputs=encoder_outputs,
            mask=src_mask,
        )

        layer_input = torch.cat([embedded, context], dim=-1)
        next_hidden_states = []

        for layer_idx, cell in enumerate(self.cells):
            next_hidden = cell(layer_input, hidden_states[layer_idx])
            next_hidden_states.append(next_hidden)
            layer_input = self.dropout(next_hidden)

        stacked_hidden = torch.stack(next_hidden_states, dim=0)
        logits = self.output_layer(torch.cat([stacked_hidden[-1], context, embedded], dim=-1))
        return logits, stacked_hidden, attention_weights


class Seq2Seq(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
    ) -> None:
        super().__init__()
        self.config = config
        self.src_pad_idx = src_vocab.pad_idx
        self.tgt_pad_idx = tgt_vocab.pad_idx
        self.tgt_sos_idx = tgt_vocab.sos_idx
        self.tgt_eos_idx = tgt_vocab.eos_idx

        self.encoder = Encoder(
            vocab_size=len(src_vocab),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.encoder_hidden_dim,
            decoder_hidden_dim=config.decoder_hidden_dim,
            num_layers=config.encoder_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
            pad_idx=src_vocab.pad_idx,
        )
        self.decoder = Decoder(
            vocab_size=len(tgt_vocab),
            embedding_dim=config.embedding_dim,
            decoder_hidden_dim=config.decoder_hidden_dim,
            encoder_output_dim=self.encoder.output_dim,
            attention_dim=config.attention_dim,
            num_layers=config.decoder_layers,
            dropout=config.dropout,
            pad_idx=tgt_vocab.pad_idx,
        )

    def _match_decoder_layers(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        if encoder_hidden.size(0) == self.decoder.num_layers:
            return encoder_hidden
        if encoder_hidden.size(0) > self.decoder.num_layers:
            return encoder_hidden[-self.decoder.num_layers :]
        missing_layers = self.decoder.num_layers - encoder_hidden.size(0)
        repeated = encoder_hidden[-1:].repeat(missing_layers, 1, 1)
        return torch.cat([encoder_hidden, repeated], dim=0)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = DEFAULT_TEACHER_FORCING,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        hidden_states = self._match_decoder_layers(encoder_hidden)
        src_mask = src.ne(self.src_pad_idx)

        input_tokens = tgt[:, 0]
        outputs = []
        attention_maps = []

        for step_idx in range(1, tgt.size(1)):
            logits, hidden_states, attention_weights = self.decoder(
                input_tokens=input_tokens,
                hidden_states=hidden_states,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
            )
            outputs.append(logits.unsqueeze(1))
            attention_maps.append(attention_weights.unsqueeze(1))

            teacher_force = random.random() < teacher_forcing_ratio
            predicted_tokens = logits.argmax(dim=-1)
            input_tokens = tgt[:, step_idx] if teacher_force else predicted_tokens

        return torch.cat(outputs, dim=1), torch.cat(attention_maps, dim=1)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int,
    ) -> list[list[int]]:
        self.eval()
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        hidden_states = self._match_decoder_layers(encoder_hidden)
        src_mask = src.ne(self.src_pad_idx)

        input_tokens = torch.full(
            (src.size(0),),
            fill_value=self.tgt_sos_idx,
            dtype=torch.long,
            device=src.device,
        )

        predictions = [[] for _ in range(src.size(0))]
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, hidden_states, _ = self.decoder(
                input_tokens=input_tokens,
                hidden_states=hidden_states,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
            )
            input_tokens = logits.argmax(dim=-1)

            for batch_idx, token_id in enumerate(input_tokens.tolist()):
                if finished[batch_idx]:
                    continue
                if token_id == self.tgt_eos_idx:
                    finished[batch_idx] = True
                    continue
                predictions[batch_idx].append(token_id)

            if finished.all():
                break

        return predictions


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_epoch(
    model: Seq2Seq,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for batch in loader:
        src = batch["src"].to(device)
        src_lengths = batch["src_lengths"].to(device)
        tgt = batch["tgt"].to(device)

        with torch.set_grad_enabled(train):
            logits, _ = model(
                src=src,
                src_lengths=src_lengths,
                tgt=tgt,
                teacher_forcing_ratio=teacher_forcing_ratio if train else 0.0,
            )
            targets = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model: Seq2Seq,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tgt_pad_idx)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,
            train=True,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=0.0,
            train=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"[{model.config.name}] Epoch {epoch:02d}/{epochs} "
            f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    return history


@torch.no_grad()
def show_sample_translations(
    model: Seq2Seq,
    dataset: TranslationDataset,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    num_examples: int = 5,
) -> None:
    model.eval()
    num_examples = min(num_examples, len(dataset))
    sampled_indices = random.sample(range(len(dataset)), k=num_examples)

    print(f"\nExemples de traductions pour {model.config.name}:")
    for idx in sampled_indices:
        src_tokens, tgt_tokens = dataset.pairs[idx]
        src_tensor = torch.tensor(
            [src_vocab.encode(src_tokens, add_eos=True)],
            dtype=torch.long,
            device=device,
        )
        src_lengths = torch.tensor([src_tensor.size(1)], dtype=torch.long, device=device)
        predicted_ids = model.greedy_decode(src_tensor, src_lengths, max_len=dataset_max_target_length(dataset))[0]

        source_text = " ".join(src_tokens)
        target_text = " ".join(tgt_tokens)
        predicted_text = tgt_vocab.decode(predicted_ids)

        print(f"EN : {source_text}")
        print(f"FR vrai : {target_text}")
        print(f"FR pred : {predicted_text}\n")


def dataset_max_target_length(dataset: TranslationDataset) -> int:
    return max(len(tgt_tokens) for _, tgt_tokens in dataset.pairs) + 2


def plot_histories(histories: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for model_name, history in histories.items():
        plt.plot(history["train_loss"], marker="o", label=f"{model_name} - train")
        plt.plot(history["val_loss"], marker="s", linestyle="--", label=f"{model_name} - val")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.title("Comparaison des losses - TP4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Courbe sauvegardee : {output_path}")


def prepare_dataloaders(
    batch_size: int,
    max_pairs: int,
    max_length: int,
    smoke_test: bool,
) -> tuple[TranslationDataset, TranslationDataset, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    if smoke_test:
        pairs = build_toy_pairs()
        print("Mode smoke test : corpus jouet charge localement.")
    else:
        corpus_path = download_and_extract_dataset(DATA_DIR)
        print(f"Corpus detecte : {corpus_path}")
        pairs = load_parallel_corpus(
            corpus_path=corpus_path,
            max_pairs=max_pairs,
            max_length=max_length,
        )

    train_pairs, val_pairs = split_pairs(pairs)

    src_vocab = Vocabulary(min_freq=MIN_FREQ)
    tgt_vocab = Vocabulary(min_freq=MIN_FREQ)
    src_vocab.build([src for src, _ in train_pairs])
    tgt_vocab.build([tgt for _, tgt in train_pairs])

    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab)

    collate_fn = make_collate_fn(src_vocab.pad_idx, tgt_vocab.pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Train pairs : {len(train_dataset)} | Val pairs : {len(val_dataset)}")
    print(f"Vocab EN : {len(src_vocab)} tokens | Vocab FR : {len(tgt_vocab)} tokens")
    return train_dataset, val_dataset, train_loader, val_loader, src_vocab, tgt_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="TP4 - Seq2Seq avec attention en PyTorch")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--teacher-forcing", type=float, default=DEFAULT_TEACHER_FORCING)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilise : {device}")

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        src_vocab,
        tgt_vocab,
    ) = prepare_dataloaders(
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        max_length=args.max_length,
        smoke_test=args.smoke_test,
    )

    baseline_config = ModelConfig(
        name="baseline_gru",
        encoder_layers=1,
        decoder_layers=1,
        bidirectional=False,
        dropout=0.2,
    )
    improved_config = ModelConfig(
        name="bidir_2layers",
        encoder_layers=2,
        decoder_layers=2,
        bidirectional=True,
        dropout=0.3,
    )

    histories: dict[str, dict[str, list[float]]] = {}

    for config in [baseline_config, improved_config]:
        print(f"\n===== Entrainement du modele : {config.name} =====")
        model = Seq2Seq(config=config, src_vocab=src_vocab, tgt_vocab=tgt_vocab).to(device)
        print(f"Nombre de parametres : {count_parameters(model):,}")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            device=device,
            teacher_forcing_ratio=args.teacher_forcing,
        )
        histories[config.name] = history
        show_sample_translations(
            model=model,
            dataset=val_dataset,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            num_examples=5 if not args.smoke_test else 3,
        )

    plot_histories(histories, Path("tp4_loss_comparison.png"))

    print("\nObservations attendues :")
    print("- L'attention aide le decodeur a se concentrer sur les mots source utiles a chaque pas.")
    print("- Le modele multi-couche/bidirectionnel converge souvent mieux, mais coute plus cher a entrainer.")
    print("- Si le modele sur-apprend ou stagne, reduis la taille du corpus ou augmente le nombre d'epoques progressivement.")


if __name__ == "__main__":
    main()
