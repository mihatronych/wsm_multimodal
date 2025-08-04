from glob import glob
from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix import YouTube
from pytubefix.innertube import _default_clients

# Download
def download_sets(wsm_sets):
    for set_csv_path in wsm_sets:
    	csv_dir = os.path.dirname(os.path.dirname(set_csv_path))
    	path = Path(set_csv_path)
    	dirname = path.parent
    	set_dir = path.stem
    	if not os.path.exists(f"{dirname}/{set_dir}"):
    		os.mkdir(f"{dirname}/{set_dir}")
    	set_df = pd.read_csv(set_csv_path)
    	video_ids = list(set_df["video_id"])
    	for v_id in video_ids:
    		try:
    			yt = YouTube(f'http://youtube.com/watch?v={v_id}', on_progress_callback = on_progress, use_oauth=True,
            allow_oauth_cache=True)
    			yt.streams.get_highest_resolution().download(output_path=f"{dirname}/{set_dir}/{v_id}")
    		except Exception as e:
    			print(dirname, set_dir, v_id)
    			print(e)