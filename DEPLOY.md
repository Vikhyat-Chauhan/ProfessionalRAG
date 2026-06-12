# Deploying ProfessionalRAG

Serverless deployment on AWS: two scale-to-zero Lambdas (query/chat + S3-triggered
ingest), embedding/reranking on Voyage AI, Pinecone as the vector store, optional
CloudFront custom domain + WAF. Infrastructure is Terraform (`infra/`); pushes to
`main` auto-deploy via GitHub Actions.

> Deeper infra reference (resource layout, secret format, follow-ups): [`infra/README.md`](infra/README.md).

---

## 0. Prerequisites (one-time, local)

```bash
brew install terraform awscli
# Docker Desktop: https://www.docker.com/products/docker-desktop  (then launch it)
aws configure        # access key / secret / region us-east-1
```

Verify:
```bash
terraform version && docker info >/dev/null && aws sts get-caller-identity
```

API keys you'll need (go straight into Secrets Manager in step 5, never into git):
**`ANTHROPIC_API_KEY`**, **`PINECONE_API_KEY`**, **`VOYAGE_API_KEY`** (voyageai.com → API Keys).

---

## 1. Bootstrap the registry + CI role

The Lambdas need images that don't exist yet, so create only ECR + the CI role first.
The placeholder image URIs are unused by this targeted apply:

```bash
cd infra
terraform init
terraform apply \
  -target=aws_ecr_repository.query \
  -target=aws_ecr_repository.ingest \
  -target=aws_iam_role.ci_deploy \
  -var query_image_uri=bootstrap -var ingest_image_uri=bootstrap
```

## 2. Build & push both images

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REG=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REG

cd ..   # repo root
docker build --platform linux/amd64 -f Dockerfile.lambda -t $REG/professionalrag-query:latest  .
docker build --platform linux/amd64 -f Dockerfile.ingest -t $REG/professionalrag-ingest:latest .
docker push $REG/professionalrag-query:latest
docker push $REG/professionalrag-ingest:latest
```

`--platform linux/amd64` is required on Apple Silicon — Lambda runs x86_64.

## 3. Write `infra/terraform.tfvars`

(Gitignored.) Fill in the image URIs from step 2 and pick your options:

```hcl
query_image_uri  = "<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/professionalrag-query:latest"
ingest_image_uri = "<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/professionalrag-ingest:latest"

alarm_email = "you@example.com"             # confirms an SNS subscription email

# Custom domain (optional — omit this block to skip CloudFront/ACM/lockdown)
domain_name     = "chat.vikhyatchauhan.com"
route53_zone_id = "Z0123456789ABC"          # leave "" if your DNS is elsewhere

# WAF (optional)
enable_waf     = true
waf_rate_limit = 2000
```

`route53_zone_id`: Route 53 console → hosted zone for your domain → Zone ID. If DNS
is on Cloudflare/Namecheap/etc., leave it `""` and use the manual path in step 4.

## 4. Full apply

```bash
cd infra
terraform apply
```

- **Route 53 path:** runs straight through — ACM validates and the alias record is created.
- **Non-Route 53 path:** the apply **pauses** on cert validation. In another shell run
  `terraform output acm_validation_records`, add those CNAME(s) at your DNS host; the apply
  continues once they resolve. Then create a CNAME from your domain → `terraform output cloudfront_domain`.

Confirm the SNS subscription email AWS sends (enables the alarms).

## 5. Populate the secret

Nothing here is echoed or committed (`infra/secret.json` is gitignored):

```bash
cd infra
cat > secret.json   # paste the JSON below, then Ctrl-D
{
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "PINECONE_API_KEY": "pcsk_...",
  "VOYAGE_API_KEY": "pa-...",
  "ProfessionalRAG_KEY": "<generate a NEW token — rotate the old EC2 one>",
  "VISIT_SALT": "<random string>"
}
aws secretsmanager put-secret-value --secret-id professionalrag/app --secret-string file://secret.json
rm secret.json
```

## 6. Re-embed the corpus (required — dimension changed 768 → 1024)

`voyage-3` is 1024-dim; the old BGE index was 768. Delete the old index in the Pinecone
console (so `store.py` recreates it at 1024), then drop the docs into S3 — the ingest
Lambda re-embeds via Voyage and upserts:

```bash
aws s3 cp ./your-docs/ s3://professionalrag-docs/ --recursive
```

## 7. Smoke test — against the domain, not the raw Function URL

With a domain configured the Function URL is IAM-locked (raw `*.lambda-url` → 403); all
traffic goes through CloudFront.

```bash
BASE=https://chat.vikhyatchauhan.com        # or: https://$(cd infra && terraform output -raw cloudfront_domain)
curl $BASE/health                            # corpus size > 0 after step 6
curl -X POST $BASE/query -H "Authorization: Bearer <ProfessionalRAG_KEY>" \
     -H "Content-Type: application/json" -d '{"question":"what does this repo do?"}'
```

New CloudFront distributions take ~5–15 min to finish deploying globally.

## 8. Wire auto-deploy & decommission EC2

```bash
cd infra && terraform output ci_deploy_role_arn
```

- GitHub → repo **Settings → Secrets and variables → Actions** → add `AWS_DEPLOY_ROLE_ARN`
  = that ARN. Every push to `main` now builds, pushes, and rolls both Lambdas
  (`.github/workflows/deploy.yml`).
- Point the site's chat widget at `https://chat.vikhyatchauhan.com`.
- Once everything works end-to-end: **terminate the EC2 instance**, delete its security
  group, and remove the old SSM `/professionalrag/api-key` parameter.

---

## Day-2 notes

- **Updates to code** → just push to `main`; CI redeploys. No manual Terraform unless infra changes.
- **Infra changes** → edit `infra/*.tf`, run `terraform fmt && terraform validate && terraform apply`.
- **Rotate a secret** → `aws secretsmanager put-secret-value ...` again; Lambdas pick it up on the next cold start.
- **Alarms** → query Lambda errors, query p95 latency, ingest errors, and (with WAF) blocked-request spikes all page the `alarm_email` SNS topic.
- **Tear everything down** → `cd infra && terraform destroy` (empty the S3 bucket first).
