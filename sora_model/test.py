from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LoadCredentialsFile("credentials.json")
# gauth.CommandLineAuth() #透過授權碼認證
drive = GoogleDrive(gauth)

try:
    file1 = drive.CreateFile({"title":'Hello.txt',"parents": [{"kind": "drive#fileLink", "id": "1YEmiqfq0LUnFAXRjpWpbwdtu0H6misgq"}]})
    file1.SetContentFile('./EEW.py')
    file1.Upload() #檔案上傳
    print("Uploading succeeded!")
    
except Exception as ex:
    print("Uploading failed.")
    print(ex)