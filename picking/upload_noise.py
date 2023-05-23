import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--month', type=int)
    parser.add_argument('--week', type=int)

    opt = parser.parse_args()
     
    return opt

if __name__ == "__main__":
    opt = parse_args()

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")
    drive = GoogleDrive(gauth)

    plot_dir = './plot'
    trace_dir = './trace'

    # get the folder of plot that is checked
    if opt.month < 10:
        week_data_plot = f"{plot_dir}/{opt.year}_0{opt.month}_week{opt.week}"
        week_data_trace = f"{trace_dir}/{opt.year}_0{opt.month}_week{opt.week}"
    else:
        week_data_plot = f"{plot_dir}/{opt.year}_{opt.month}_week{opt.week}"
        week_data_trace = f"{trace_dir}/{opt.year}_{opt.month}_week{opt.week}"

    noise_files = os.listdir(week_data_plot)
    try:
        for f in noise_files:
            if f.split('.')[-1] != 'png':
                continue

            trace_noise = os.path.join(week_data_trace, f.split('.png')[0] + '.pt')

            file1 = drive.CreateFile({"title":f.split('.png')[0] + '.pt',"parents": [{"kind": "drive#fileLink", "id": "10fZP97Lnv259ki_mLX-Ibu9d85mnPk6J"}]})
            file1.SetContentFile(trace_noise)
            file1.Upload() 

            print("Upload successful -> ", trace_noise)
            os.remove(trace_noise)

        shutil.rmtree(week_data_plot)
        shutil.rmtree(week_data_trace)
    except Exception as e:
        print(e)
        print("Upload failed")

        
