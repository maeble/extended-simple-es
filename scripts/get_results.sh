# ---------- rename files for zipping process
find . -name "*:*" -exec rename -n 's|:|-|g' {} \;
find . -name "*:*" -exec rename 's|:|-|g' {} \;

# ---------- collect gifs, test_returns
#./scripts/test_run.sh # TODO

# ---------- zip outputs
#./scripts/get/get_gifs.sh
zip -r es.zip logs/ # saved_models, results
