import smtplib

from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import time

fromaddr = "teslak40@gmail.com"
frompsw= ""
toaddr = "milana.diletta@gmail.com"
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
a=1
b=2
msg['Subject'] = '[{}, {}] Progress report - Training on Tesla'.format(a,b)

body = "TIME LEFT: approx 5 hours and 25 minutes.\nHello, this is a report from your beloved Tesla machine. Enjoy!"
msg.attach(MIMEText(body, 'plain'))

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, frompsw)
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()

print("done")
