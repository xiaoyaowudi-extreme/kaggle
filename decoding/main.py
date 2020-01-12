#!/usr/bin/python

import numpy as np
import argparse
import cv2
import os

methods={}

#--idx--
import idx
methods['idx1']=idx.idx_decode.decode_idx1_ubyte
methods['idx3']=idx.idx_decode.decode_idx3_ubyte
#--idx--

if __name__=='__main__':
	args=argparse.ArgumentParser(prog="decode")
	args.add_argument(
						"type",
						choices=methods.keys(),
						type=str,
						help='file type'
						)
	args.add_argument(
						"-f",
						"--file",
						required=True,
						type=str,
						help='file path'
						)
	args.add_argument(
						"-d",
						"--directory",
						required=True,
						type=str,
						help='output directory path'
						)
	args.add_argument(
						"-s",
						"--suffix",
						default='.jpg',
						type=str,
						help='the suffix of output file'
						)
	arg          = args.parse_args()
	function     = methods[arg.type]
	images       = function(arg.file)
	image_suffix = ['.jpg', '.png', '.bmp', 'tiff', '.jpeg']
	if arg.suffix in image_suffix:
		for i in range(images.shape[0]):
			cv2.imwrite(os.path.join(arg.directory,str(i)+arg.suffix),images[i])
	else:
		for i in range(images.shape[0]):
			with open(os.path.join(arg.directory,str(i)+arg.suffix),'w') as f:
				f.write(str(images[i]))
	
