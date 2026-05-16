# S3 → Lambda → /ingest setup (AWS console walkthrough)

End state: drop a file into an S3 bucket → it appears in `/chat` answers a minute later, no SSH, no manual `curl`.

Assumes the EC2 service is already running at `http://<EC2_PUBLIC_IP>:8080` with an instance role named `InstanceRole` (the one with DynamoDB permissions).

All steps are in **us-east-1**. If you use a different region, the EC2 instance, S3 bucket, Lambda, and SSM parameter must all be in the same one.

---

## 1. Create the S3 bucket

1. **S3 console** → **Create bucket**
2. Name: `professionalrag-docs` (must be globally unique — add a suffix if taken)
3. Region: **us-east-1**
4. Block all public access: **on** (default)
5. Versioning: **Enable** (optional but recommended — lets you roll back a bad upload)
6. Leave everything else default → **Create bucket**

---

## 2. Store the API key in SSM Parameter Store

1. **Systems Manager** → **Parameter Store** → **Create parameter**
2. Name: `/professionalrag/api-key`
3. Tier: **Standard**
4. Type: **SecureString**
5. KMS key source: **My current account → alias/aws/ssm** (default)
6. Value: paste the value of `ProfessionalRAG_KEY` from your `.env` on the EC2 box
7. **Create parameter**

---

## 3. Grant the EC2 instance role `s3:GetObject` on the bucket

The EC2 container needs to download objects when `/ingest` fires.

1. **IAM** → **Roles** → **InstanceRole** → **Add permissions** → **Create inline policy** → **JSON**
2. Paste, replacing `professionalrag-docs` with your bucket name if different:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["s3:GetObject", "s3:HeadObject"],
       "Resource": "arn:aws:s3:::professionalrag-docs/*"
     }]
   }
   ```

3. Name: `S3ReadProfessionalRAGDocs` → **Create policy**

No EC2 restart needed — IAM picks this up within seconds.

---

## 4. Create the Lambda function

1. **Lambda** → **Create function** → **Author from scratch**
2. Name: `professionalrag-ingest-trigger`
3. Runtime: **Python 3.12**
4. Architecture: **x86_64**
5. Execution role: **Create a new role with basic Lambda permissions**
6. **Create function**

### 4a. Upload the code

In the function's **Code** tab:

1. Open `lambda/ingest_trigger.py` from this repo, copy its contents
2. In the Lambda console, replace `lambda_function.py` content with what you copied
3. **Rename the file** in the Lambda console: right-click `lambda_function.py` → Rename → `ingest_trigger.py`
4. **Deploy** (the orange button)

### 4b. Set the handler

**Configuration** → **General configuration** → **Edit**:

- Handler: `ingest_trigger.handler`
- Timeout: **30 sec**
- Memory: **128 MB**
- **Save**

### 4c. Set environment variables

**Configuration** → **Environment variables** → **Edit** → **Add**:

| Key | Value |
|---|---|
| `INGEST_URL` | `http://<EC2_PUBLIC_IP>:8080/ingest` |
| `API_KEY_PARAM` | `/professionalrag/api-key` |

**Save**.

> ⚠️ The Lambda calls the EC2 over the public internet here, which means anyone scanning that IP can also call `/ingest`. That's fine because the Bearer token gates it. If you put nginx + HTTPS in front later, change `INGEST_URL` to `https://api.yourdomain.com/ingest`.

### 4d. Give Lambda permission to read the SSM parameter

**Configuration** → **Permissions** → click the role name (opens IAM in a new tab) → **Add permissions** → **Create inline policy** → **JSON**:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "ssm:GetParameter",
    "Resource": "arn:aws:ssm:us-east-1:*:parameter/professionalrag/api-key"
  }]
}
```

Name: `ReadProfessionalRAGApiKey` → **Create policy**.

---

## 5. Wire S3 → Lambda

1. **S3 console** → bucket `professionalrag-docs` → **Properties** tab
2. Scroll to **Event notifications** → **Create event notification**
3. Name: `ingest-on-upload`
4. Prefix (optional): `docs/` — limits triggers to keys under `docs/`. Leave blank to trigger on every upload.
5. Suffix (optional): blank, or restrict to e.g. `.pdf`
6. Event types: check **All object create events** (`s3:ObjectCreated:*`)
7. Destination: **Lambda function** → choose `professionalrag-ingest-trigger`
8. **Save changes**

S3 will warn you that it's adding an invocation permission to the Lambda — accept it. This is just S3 telling Lambda "I'm allowed to call you."

---

## 6. Test it end-to-end

```bash
# From your laptop (with AWS CLI configured), or any machine with credentials:
echo "Llamas are pack animals from the Andes." > llama-facts.txt
aws s3 cp llama-facts.txt s3://professionalrag-docs/docs/llama-facts.txt
```

Within ~5 seconds:

1. **Lambda console** → `professionalrag-ingest-trigger` → **Monitor** → **View CloudWatch logs**
   You should see: `Ingest dispatched: s3://professionalrag-docs/docs/llama-facts.txt → HTTP 200`

2. On the EC2 box:
   ```bash
   docker logs rag --tail 20
   # Should show: "Downloading s3://...", "Created N chunks", "Ingested N chunks"
   ```

3. Test the new knowledge:
   ```bash
   KEY=$(grep ProfessionalRAG_KEY .env | cut -d= -f2)
   curl -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"question":"where are llamas from?"}' \
     http://localhost:8080/query
   ```

   The answer should reference the Andes.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Lambda logs show `urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>` | EC2 security group blocking inbound 8080 from Lambda's egress | Open 8080 to `0.0.0.0/0` in the SG (or move both into a VPC) |
| Lambda logs `HTTP 401` | Wrong API key in SSM parameter | Verify `aws ssm get-parameter --name /professionalrag/api-key --with-decryption` matches `.env` |
| Lambda logs `botocore... ParameterNotFound` | Wrong parameter name or wrong region | Lambda + SSM must be in same region; double-check `API_KEY_PARAM` env var |
| EC2 logs `NoSuchKey` or `AccessDenied` from S3 | EC2 role lacks `s3:GetObject` | Re-check step 3 |
| Nothing happens after upload | S3 event filter excluded your key | Remove the prefix/suffix filter; try again |
| `/chat` doesn't find the new doc | Embedding still running, or chunk count too small | `docker logs rag` — look for "Ingested N chunks"; query takes ~30s after dispatch on first ingest |

---

## What this costs

| Resource | Free tier | Beyond free tier |
|---|---|---|
| S3 storage | 5 GB | $0.023/GB/month |
| S3 requests | 2k PUTs, 20k GETs/month | $0.005/1k PUTs |
| Lambda | 1M invocations + 400k GB-sec/month | $0.20/1M invocations |
| SSM Standard parameters | unlimited | free |
| CloudWatch logs | 5 GB ingestion | $0.50/GB |

For "I drop a few files per week" this is effectively free.

---

## Future: handling deletions

S3 also fires `s3:ObjectRemoved:*` events. The `/ingest` endpoint doesn't have a deletion counterpart yet — adding one would mean:

1. New endpoint `POST /forget` accepting `{"source": "s3://..."}`
2. Calls `VectorStore._remove_source(_source_tag(s3_fingerprint(source)))`
3. Update the Lambda to also handle `ObjectRemoved:*` events

Worth doing once the corpus matters enough that stale chunks are a real cost.
