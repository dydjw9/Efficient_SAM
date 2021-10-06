
import smtplib
import base64
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.utils import COMMASPACE
import socket

SENDER = 'anonymousttnt@gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
USER_ACCOUNT = {'username':'anonymousttnt@gmail.com', 'password':'11227788aa'}
SUBJECT = "Test Test"


def send_email(text,subject = socket.gethostname()+" reports",receivers=["anonymousttnt@gmail.com","dydjw9@gmail.com"], sender=SENDER, user_account=USER_ACCOUNT):
    msg_root = MIMEMultipart()  # 创建一个带附件的实例
    msg_root['Subject'] = subject  # 邮件主题
    msg_root['To'] = COMMASPACE.join(receivers)  # 接收者
    msg_text = MIMEText(text, 'html', 'utf-8')  # 邮件正文
    msg_root.attach(msg_text)  # attach邮件正文内容

    smtp = smtplib.SMTP('smtp.gmail.com:587')
    smtp.ehlo()
    smtp.starttls()
    smtp.login(user_account['username'], user_account['password'])
    smtp.sendmail(sender, receivers, msg_root.as_string())

if __name__=="__main__":
    send_email("test")

