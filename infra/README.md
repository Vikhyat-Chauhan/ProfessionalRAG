# Infrastructure (Terraform)

Serverless deployment of ProfessionalRAG on AWS Lambda — replaces the manual EC2
box. Two container-image Lambdas (query/chat + S3-triggered ingest), a Function
URL with response streaming, DynamoDB, S3, Secrets Manager, CloudWatch alarms,
and a GitHub OIDC deploy role.

## Layout

| File | Resources |
|------|-----------|
| `main.tf` | provider, ECR repos, DynamoDB `visits`, docs S3 bucket, app secret |
| `lambda.tf` | query + ingest Lambdas, execution role, Function URL, S3 notification |
| `cicd.tf` | GitHub OIDC provider + `ci-deploy` role (ECR push + Lambda update) |
| `domain.tf` | (optional) ACM cert + CloudFront in front of the Function URL, Route 53 records, rate-based WAF |
| `observability.tf` | SNS alerts, log groups (30-day retention), error/latency alarms |
| `outputs.tf` | Function URL, ECR repos, deploy-role ARN, secret ARN |

## Bootstrap (chicken-and-egg: Lambdas need images, images need ECR)

```bash
cd infra
terraform init

# 1. Create ECR repos + the CI role first (targeted apply).
terraform apply -target=aws_ecr_repository.query -target=aws_ecr_repository.ingest \
                -target=aws_iam_role.ci_deploy

# 2. Build & push initial images (or just run the Deploy workflow once).
#    See the root Deploy workflow; locally you can do the docker build/push by hand.

# 3. Full apply, passing the pushed image URIs.
terraform apply \
  -var "query_image_uri=<acct>.dkr.ecr.us-east-1.amazonaws.com/professionalrag-query:latest" \
  -var "ingest_image_uri=<acct>.dkr.ecr.us-east-1.amazonaws.com/professionalrag-ingest:latest"
```

Then set the `AWS_DEPLOY_ROLE_ARN` GitHub Actions secret to `terraform output ci_deploy_role_arn`.
After that, pushes to `main` build, push, and roll both functions automatically.

## Secret format

The `professionalrag/app` secret is a single JSON blob (populate via console/CLI,
never in Terraform state). `config.py` hydrates these into env vars at cold start
when `SECRETS_ARN` is set:

```json
{
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "PINECONE_API_KEY": "pcsk_...",
  "VOYAGE_API_KEY": "pa-...",
  "ProfessionalRAG_KEY": "<rotated bearer token>",
  "VISIT_SALT": "<random>"
}
```

```bash
aws secretsmanager put-secret-value --secret-id professionalrag/app \
  --secret-string file://secret.json
```

## Remaining manual / follow-up steps

- **Rotate `ProfessionalRAG_KEY`** — the old value lived in plaintext `.env` on EC2.
  Generate a fresh token, store it in the secret, decommission the EC2 instance.
- **Custom domain (codified in `domain.tf`)** — set `domain_name` to enable CloudFront +
  ACM in front of the Function URL, then point the site's chat widget at it. Enabling a
  domain also flips the Function URL to **AWS_IAM** auth and signs CloudFront→Lambda
  requests with **Origin Access Control**, so the raw `*.lambda-url.on.aws` host returns
  403 — all traffic must go through CloudFront (and still carries the app's Bearer auth).
  Smoke-test against the domain / `cloudfront_domain`, not the Function URL. Two paths:
  - *Route 53*: also set `route53_zone_id`; cert validation and the alias record are created
    automatically — a single `terraform apply` brings the domain up.
  - *Other DNS host*: leave `route53_zone_id` empty. `apply` blocks while the cert validates —
    add the records from the `acm_validation_records` output at your DNS provider, then once
    it completes, create a CNAME from your domain to the `cloudfront_domain` output.
- **Abuse protection** — set `enable_waf = true` to attach a rate-based WAF (`waf_rate_limit`
  req / 5 min / IP) to CloudFront. `slowapi`'s in-memory limiter is per-instance and
  near-useless on Lambda, so the WAF is the real throttle. Enabling it also wires a
  CloudWatch alarm (`waf_blocked_alarm_threshold` blocks / 5 min → the alerts SNS topic)
  so you're paged when blocking kicks in.
- **One-time corpus migration** — the embedding model changed (768→1024 dim). Recreate
  the Pinecone index and re-ingest: drop the docs back into the S3 bucket (the ingest
  Lambda re-embeds via the hosted model) or run `python cli.py ingest <source>` locally.
- **Sentry (optional)** — add the Sentry SDK around the FastAPI handlers and set
  `SENTRY_DSN` in the secret for error tracking beyond CloudWatch.

## Migrating off EC2

Once the Function URL serves `/health`, `/query`, and `/chat` and the ingest Lambda
upserts on S3 upload, the EC2 instance, its security group, and the old SSM
`/professionalrag/api-key` parameter can be deleted. The legacy console-driven setup
in `docs/aws-s3-trigger-setup.md` is superseded by this Terraform.
