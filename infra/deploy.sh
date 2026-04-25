#!/usr/bin/env bash
set -euo pipefail

REGION="us-east-2"
ACCOUNT_ID="901059153423"
INSTANCE_ID="i-08b088fd29cbd5434"
ELASTIC_IP="3.12.13.119"
FUNCTION_NAME="poet-wake"
ROLE_NAME="poet-wake-lambda-role"
API_NAME="poet-wake-api"

echo "=== 1. Create IAM role for Lambda ==="

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

aws iam create-role \
  --role-name "$ROLE_NAME" \
  --assume-role-policy-document "$TRUST_POLICY" \
  2>/dev/null || echo "Role already exists, continuing..."

INLINE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ec2:DescribeInstances",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "ec2:StartInstances",
      "Resource": "arn:aws:ec2:${REGION}:${ACCOUNT_ID}:instance/${INSTANCE_ID}"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name poet-wake-ec2 \
  --policy-document "$INLINE_POLICY"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "Role ARN: $ROLE_ARN"

echo "Waiting for role to propagate..."
sleep 10

echo "=== 2. Package and create Lambda function ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/lambda_wake"
zip -j /tmp/poet-wake.zip handler.py

aws lambda create-function \
  --region "$REGION" \
  --function-name "$FUNCTION_NAME" \
  --runtime python3.12 \
  --handler handler.lambda_handler \
  --role "$ROLE_ARN" \
  --zip-file fileb:///tmp/poet-wake.zip \
  --timeout 10 \
  --memory-size 128 \
  --environment "Variables={INSTANCE_ID=${INSTANCE_ID},ELASTIC_IP=${ELASTIC_IP}}" \
  2>/dev/null || {
    echo "Function exists, updating code..."
    aws lambda update-function-code \
      --region "$REGION" \
      --function-name "$FUNCTION_NAME" \
      --zip-file fileb:///tmp/poet-wake.zip
    aws lambda update-function-configuration \
      --region "$REGION" \
      --function-name "$FUNCTION_NAME" \
      --environment "Variables={INSTANCE_ID=${INSTANCE_ID},ELASTIC_IP=${ELASTIC_IP}}"
  }

echo "=== 3. Create HTTP API Gateway ==="

API_ID=$(aws apigatewayv2 create-api \
  --region "$REGION" \
  --name "$API_NAME" \
  --protocol-type HTTP \
  --query ApiId --output text 2>/dev/null) || {
    API_ID=$(aws apigatewayv2 get-apis \
      --region "$REGION" \
      --query "Items[?Name=='${API_NAME}'].ApiId | [0]" --output text)
    echo "API already exists: $API_ID"
  }

INTEGRATION_ID=$(aws apigatewayv2 create-integration \
  --region "$REGION" \
  --api-id "$API_ID" \
  --integration-type AWS_PROXY \
  --integration-uri "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}" \
  --payload-format-version "2.0" \
  --query IntegrationId --output text)

aws apigatewayv2 create-route \
  --region "$REGION" \
  --api-id "$API_ID" \
  --route-key 'GET /' \
  --target "integrations/${INTEGRATION_ID}"

aws apigatewayv2 create-stage \
  --region "$REGION" \
  --api-id "$API_ID" \
  --stage-name '$default' \
  --auto-deploy 2>/dev/null || echo "Stage already exists"

echo "=== 4. Grant API Gateway permission to invoke Lambda ==="

aws lambda add-permission \
  --region "$REGION" \
  --function-name "$FUNCTION_NAME" \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*" \
  2>/dev/null || echo "Permission already exists"

API_URL="https://${API_ID}.execute-api.${REGION}.amazonaws.com"
echo ""
echo "=== Done ==="
echo "Gateway URL: ${API_URL}"
echo "This URL will wake the EC2 instance and redirect to Gradio."
