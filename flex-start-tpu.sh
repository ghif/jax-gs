gcloud alpha compute tpus queued-resources create tpu-southamerica-queue \
    --zone=southamerica-east1-c \
    --accelerator-type=v6e-4 \
    --runtime-version=v2-alpha-tpuv6e \
    --node-id=my-tpu-node \
    --provisioning-model=flex-start \
    --max-run-duration=96h \
    --valid-until-duration=96h \
    --labels=purpose=flex-start # Fix to 