#!/bin/zsh

docker run \
	-it \
	--privileged \
	--device=/dev/video0:/dev/video0 \
	deb-py-3.9 \
	/bin/bash

