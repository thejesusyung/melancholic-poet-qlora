import json
import os

import boto3

INSTANCE_ID = os.environ["INSTANCE_ID"]
ELASTIC_IP = os.environ["ELASTIC_IP"]
GRADIO_PORT = os.environ.get("GRADIO_PORT", "7860")
REGION = os.environ.get("AWS_REGION", "us-east-2")

ec2 = boto3.client("ec2", region_name=REGION)

WARMING_HTML = f"""<!DOCTYPE html>
<html>
<head>
  <title>Melancholic Poet — Starting Up</title>
  <meta http-equiv="refresh" content="30">
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 480px; margin: 80px auto; text-align: center; color: #333; }}
    .spinner {{ display: inline-block; width: 28px; height: 28px; border: 3px solid #ccc;
      border-top-color: #555; border-radius: 50%; animation: spin 0.8s linear infinite; margin-bottom: 16px; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
  <div class="spinner"></div>
  <h2>Waking the poet&hellip;</h2>
  <p>The server is starting up. This page will refresh automatically.<br>
  It usually takes 2&ndash;3 minutes.</p>
</body>
</html>"""


def lambda_handler(event, context):
    resp = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
    state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]

    if state == "running":
        return {
            "statusCode": 302,
            "headers": {"Location": f"http://{ELASTIC_IP}:{GRADIO_PORT}"},
            "body": "",
        }

    if state == "stopped":
        ec2.start_instances(InstanceIds=[INSTANCE_ID])

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": WARMING_HTML,
    }
