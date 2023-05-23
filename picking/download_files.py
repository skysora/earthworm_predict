from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from tqdm import tqdm
import os

gauth = GoogleAuth()
gauth.LoadCredentialsFile("credentials.json")
drive = GoogleDrive(gauth)

try:
    # get picking.log 
    savedir = './history_log/pick_log/'
    pick_log_folder = '1Y2o_Pp6np8xnxl0QysU4-zVEm4mWOI6_'
    file_list = drive.ListFile({'q': f"'{pick_log_folder}' in parents and trashed=false"}).GetList()
    for file1 in tqdm(file_list, total=len(file_list)):
        if os.path.exists(file1):
            continue
        print('title: %s, id: %s' % (file1['title'], file1['id']))
        # get file and save
        file1.GetContentFile(f"{savedir}{file1['title'].split('/')[-1]}")

    # get original_picking.log 
    savedir = './history_log/original_log/'
    orignal_pick_log_folder = '1QmeQbsyjajpKHQXcuNxjNm426J--GZ15'
    file_list = drive.ListFile({'q': f"'{orignal_pick_log_folder}' in parents and trashed=false"}).GetList()
    for file1 in tqdm(file_list, total=len(file_list)):
        if os.path.exists(file1):
            continue
        print('title: %s, id: %s' % (file1['title'], file1['id']))
        # get file and save
        file1.GetContentFile(f"{savedir}{file1['title'].split('/')[-1]}")

    # get notify.log 
    savedir = './history_log/notify_log/'
    notify_log_folder = '1aqLRskDjn7Vi7WB-uzakLiBooKSe67BD'
    file_list = drive.ListFile({'q': f"'{notify_log_folder}' in parents and trashed=false"}).GetList()
    for file1 in tqdm(file_list, total=len(file_list)):
        if os.path.exists(file1):
            continue
        print('title: %s, id: %s' % (file1['title'], file1['id']))
        # get file and save
        file1.GetContentFile(f"{savedir}{file1['title'].split('/')[-1]}")

    # get pytorch tensors 
    savedir = './history_log/pt/'
    pt_folder = '1WHLH8Fn0rf72-uId_TjLimBQQL47h9ey'
    file_list = drive.ListFile({'q': f"'{pt_folder}' in parents and trashed=false"}).GetList()
    for file1 in tqdm(file_list, total=len(file_list)):
        if os.path.exists(file1):
            continue
        print('title: %s, id: %s' % (file1['title'], file1['id']))
        # get file and save
        file1.GetContentFile(f"{savedir}{file1['title']}")

except Exception as e:
    print(e)
    print('download failed')
