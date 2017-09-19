import requests
import os
import argparse
import sys
from time import sleep

import requests
from bs4 import BeautifulSoup as bs
from threading import Thread 

## variables
fotolia_download_button = 'comp-download-buttons row-large'
istock_base_download_button = 'asset-link draggable'

## get the url of the image
def _get_image_url_fotolia(base_url, minVal, directory, index=0, num_retries = 5):
	img_url = ""
	retries = 0
	while retries < num_retries:
		# try
		r = requests.get(base_url + str(minVal + index))
		if r.status_code == 200:
			soup = bs(r.content, 'lxml')
			row = soup.find_all(attrs={'class': fotolia_download_button})
			# check row
			if len(row) > 0:
				link = row[0].findChildren()[0]
				if(link.attrs.has_key('href')):
					img_url = link.attrs['href']
					__download_and_save_image(img_url, directory)
				else:
					print("Error, check: ")
					print(link)
			else:
				print("There is no image download button.")

			break
		else:
			retries += 1

	return img_url
		
# get the link
def _get_istock_page_and_download(link, directory):
	_media_url = "media.istockphoto.com"
	r = requests.get(link)
	if r.status_code == 200:
		soup = bs(r.content, 'lxml')
		img = []
		img = filter(lambda x: _media_url in x.attrs['src'], filter(lambda x: x.attrs.has_key('src'), soup.find_all('img')))
		if img == []:
			print("Cannot find image.")
		else:
			img_link = img[0].attrs['src']
			__download_and_save_image(img_link, directory, src='istock')
	else:
		print("Cannot connect to : " + link)

# download and save a given image
def __download_and_save_image(link, directory, src='fotolia'):
	print("Attempting to download: " + link)
	r = requests.get(link)
	if r.status_code == 200:
		
		# depends on source
		if src == 'fotolia':
			try:
				filename = r.headers['Content-Disposition'].split('filename="')[1][:-2]
			except:
				print("No Content-Disposition header present.")
				return
		elif src == 'istock':
			try:
				filename = r.headers['Content-Disposition'].split('filename=')[1]
			except:
				print("No Content-Disposition header present.")
				return

		filename = os.sep.join([directory, filename])
		print("Saving to filename: %s "%(filename))
		with open(filename, 'wb') as f:
			f.write(r.content)
	else:
		print("Couldn't download from link: " + link)


# function to scrape from fotolia
def fotolia_scrape(directory, minVal=137840645, n_images=100):
	# make the dir first
	if not os.path.isdir(directory):
		os.mkdir(directory)

	base_url = "https://www.fotolia.com/Content/Comp/"
	image_url_list = [] 
	index = 0

	# check thread list
	thread_list = []

	# start threads
	for index in xrange(n_images):
		th = Thread(target=_get_image_url_fotolia, args=(base_url, minVal, directory, index))
		thread_list.append(th)
		th.start()

	# join
	for th in thread_list:
		th.join()


# function to scrape from istock
def istock_scrape(directory, topic="abstract", n_images=100):

	## iStock blocks you, be careful
	# raise NotImplementedError("iStockPhotos blocks you, be careful.")

	webpage = "https://www.istockphoto.com"
	base_search_url = "http://www.istockphoto.com/in/photos/%s"%topic

	r = requests.get(base_search_url)
	links_list = []
	if r.status_code == 200:
		soup = bs(r.content, 'lxml')
		links = map(lambda x: webpage + x.attrs['href'], soup.find_all(attrs={'class': istock_base_download_button}))
		links_list += links

		nextPageLink = soup.find_all(attrs={'id':'next-gallery-page'})
		print("Moving to next page.")
		sleep(0.5)

		while(nextPageLink != [] and len(links_list) < n_images):
			href = webpage + nextPageLink[0].attrs['href']
			r = requests.get(href)
			if r.status_code == 200:
				soup = bs(r.content, 'lxml')
				links = map(lambda x: webpage + x.attrs['href'], soup.find_all(attrs={'class': istock_base_download_button}))
				links_list += links
				nextPageLink = soup.find_all(attrs={'id':'next-gallery-page'})
				print("Moving to next page.")
			else:
				nextPageLink = []
				print("No next page found.")

		thread_list = []
		## we have the list of link, go to each link and download it
		for link in links_list:
			th  = Thread(target=_get_istock_page_and_download, args=(link, directory))
			thread_list.append(th)
			th.start()
			th.join()
			sleep(1)

		# for th in thread_list:
		# 	th.join()


''' 
Main function here
'''
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Scrape from stock images')
	parser.add_argument('-f', dest='folder', help='Specify the folder where to place the images.')
	parser.add_argument('-u', dest='url', help='Specify the place from where to scrape.')
	args = parser.parse_args()

	if args.url is None:
		parser.print_help()
		sys.exit(0)
	else:
		# define the folder
		if args.folder is None:
			directory = "."
		else:
			directory = args.folder

		# check for the param
		if "fotolia" in args.url:
			fotolia_scrape(directory, n_images=100)

		elif "istock" in args.url:
			istock_scrape(directory, n_images=150, topic='mountains')


		print("Done.")

		