## Automatic watermark detection and removal
This was the project that was built as part of project for CS663 (Digital Image Processing).

### Introduction
Visible watermarking is a widely-used technique for marking and protecting copyrights of many millions of images on the web, yet it suffers from an inherent security flaw - watermarks are typically added in a consistent manner to many images. For example, they can be added to the center or corner of an image, and usually they are added with the same opacity levels, and to provide a bit of variation, are usually scaled, rotated, or applied in a periodic fashion to images. While removing a watermark from a single image can be tedious even for a person with good photoshop skills, removing watermarks from a number of images with some consistent property can be done automatically (or semi-automatically). 
This is a crude implementation of the paper