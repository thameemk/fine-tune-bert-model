import logging

import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content


def send_email_using_sendgrid(from_email, to_email, subject, content):
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))

    mail = Mail(Email(from_email), To(to_email), subject, Content("text/plain", content))

    # Get a JSON-ready representation of the Mail object
    mail_json = mail.get()

    # Send an HTTP POST request to /mail/send
    response = sg.client.mail.send.post(request_body=mail_json)

    if response.status_code == 200:
        return True
    else:
        logging.error(f"Error sending email using sendgrid - status: {response.status_code}")
        return False
