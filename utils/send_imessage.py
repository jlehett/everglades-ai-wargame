import os

def send_imessage(message, to):
    cmd = "osascript -e 'tell application \"Messages\" to send \""
    cmd += str(message)
    cmd += "\" to buddy \""
    cmd += to
    cmd += "\"'"
    os.system(cmd)