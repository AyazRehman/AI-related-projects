#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

