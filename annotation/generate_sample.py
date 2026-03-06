"""Generate sample.json by fetching items from ClimbMix and 4chan via DuckDB."""

from annotation.config import ITEMS_PER_SOURCE
from annotation.sampling import (
    draw_stratified_sample,
    load_items_from_sources,
    load_sample,
    sample_path,
    save_sample,
)


def main():
    existing = load_sample()
    if existing is not None:
        print(f"sample.json already exists with {len(existing)} items. Delete it to regenerate.")
        return

    print("Fetching items from both sources...")
    all_items = load_items_from_sources()
    climbmix_items = [i for i in all_items if i["subset"] == "climbmix"]
    fourchan_items = [i for i in all_items if i["subset"].startswith("4chan/")]
    print(f"Pool: {len(climbmix_items)} ClimbMix, {len(fourchan_items)} 4chan")

    climbmix_ids = [i["item_id"] for i in climbmix_items]
    fourchan_ids = draw_stratified_sample(
        fourchan_items, n=ITEMS_PER_SOURCE, min_per_stratum=1,
    )
    save_sample(all_items, climbmix_ids + fourchan_ids)
    saved = load_sample()
    print(f"Saved {len(saved)} items to {sample_path()}")


if __name__ == "__main__":
    main()
