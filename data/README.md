# Data

ChatPhone uses the **Amazon Cell Phones Reviews** dataset (Kaggle):
<https://www.kaggle.com/datasets/grikomsn/amazon-cell-phones-reviews>

The dataset isn't included in this repository. Download it and place these two files in this folder:

```text
data/
├── 20190928-items.csv     # one row per phone: asin, brand, title, rating, totalReviews, prices, ...
└── 20190928-reviews.csv   # one row per review: asin, rating, verified, title, body, ...
```

If the version you download has different file names (for example a newer date), rename the files or update `ITEMS_FILE` / `REVIEWS_FILE` in `search_api/search_api.py`.
