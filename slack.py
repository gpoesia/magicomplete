import urllib
import os
import json

WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL')

if WEBHOOK_URL is None:
    print('Environment variable SLACK_WEBHOOK_URL not set: Slack messages will not be sent.')

def send_message(text):
    if WEBHOOK_URL is not None:
        r = urllib.request.Request(WEBHOOK_URL,
                                   data=json.dumps({'text': text}).encode('utf-8'),
                                   headers={
                                       'Content-Type': 'application/json'
                                   },
                                   method='POST')
        with urllib.request.urlopen(r) as f:
            status = str(f.status)
    else:
        status = 'not sent - no webhook URL'

    print('Slack message: {} (status: {})'.format(text, status))
