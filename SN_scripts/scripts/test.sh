/home/toolkit/./eai job new -f SN_scripts/config/default.yaml --field id -- /bin/bash -c \
"source /opt/conda/bin/activate /home/toolkit/mteb-lite/.conda && \
bash run_all.sh NFCorpus intfloat/e5-small-v2 384 test \
>> /home/toolkit/mteb-lite/out.log 2>&1"