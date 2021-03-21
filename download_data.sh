echo "LOGGING IN"
aicrowd login --api-key $API_KEY
echo

echo "DOWNLOADING DATA"
aicrowd dataset download --challenge mabe-task-1-classical-classification
echo

echo "PREPPING DIRECTORIES"
rm -rf data
mkdir data
echo

echo "MOVING FILES"
mv train.npy data/train.npy
mv test-release.npy data/test.npy
mv sample-submission.npy data/sample_submission.npy
echo
echo "ALL SET"
