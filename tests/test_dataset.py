from nanochat.dataset import DEFAULT_BASE_URL, get_dataset_base_url


def test_dataset_base_url_defaults_to_huggingface(monkeypatch):
    monkeypatch.delenv("NANOCHAT_DATASET_BASE_URL", raising=False)

    assert get_dataset_base_url() == DEFAULT_BASE_URL


def test_dataset_base_url_supports_mirror_and_strips_slash(monkeypatch):
    monkeypatch.setenv(
        "NANOCHAT_DATASET_BASE_URL",
        "https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main/",
    )

    assert get_dataset_base_url() == (
        "https://hf-mirror.com/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
    )
