#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function, absolute_import

import sys
import os
import numpy as np
import logging
import time


from deepatari import __version__
from deepatari.tools import parse_args
from deepatari.experiments import Experiment

__author__ = "Ruben Glatt"
__copyright__ = "Ruben Glatt"
__license__ = "MIT"


def run():
    # read command line arguments and initialize logger
    args = parse_args(sys.argv[1:])
    # prepare target directory for logs, stats and nets
    if args.log_stats or not args.log_type == 'stdout':
        result_dir = "%s_%s_%s_%s" % (str(time.strftime("%Y-%m-%d_%H-%M")) ,str(args.game.lower()), str(args.learner_type.lower()), str(args.optimizer.lower()))
        target_dir = os.path.join(os.getcwd(),"results", result_dir)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        args_dump = open(os.path.join(target_dir, 'args_dump.txt'), 'w', 0)
        args_dict = vars(args)
        for key in sorted(args_dict):
            args_dump.write("%s=%s\n" % (str(key), str(args_dict[key])))
        #args_dump.write(str(args))
        args_dump.flush()
        args_dump.close()
    else:
        target_dir = None
    # make sure no loggers are already active
    try:
        logging.root.handlers.pop()
    except IndexError:
        # if no logger exist the list will be empty and we need to catch the resulting error
        pass
    if args.log_type == 'stdout':
        logging.basicConfig(level=getattr(logging, (args.log_level).upper(), None),
                        stream=sys.stdout,
                        format='[%(asctime)s][%(levelname)s][%(module)s][%(funcName)s] %(message)s')
        '''
        logging.basicConfig(level=getattr(logging, (args.log_level).upper(), None),
                            stream=sys.stdout,
                            format='[%(asctime)s] %(message)s')
        '''
    else:
        logging.basicConfig(level=getattr(logging, (args.log_level).upper(), None),
                        format='[%(asctime)s][%(levelname)s][%(module)s][%(funcName)s] %(message)s',
                        filename=os.path.join(target_dir, 'experiment.log'),
                        filemode='w')
    _logger = logging.getLogger(__name__)
    _logger.info("########## STARTING ##########")
    _logger.info("parameter: %s" % str(sys.argv[1:]))
    _logger.debug("%s" % args)

    try:
        ExperimentClass = getattr(
                __import__('deepatari.experiments.' + args.exp_type.lower(),
                        fromlist=[args.exp_type]),
                args.exp_type)
    except ImportError:
        sys.stderr.write("ERROR: missing python module: " + args.exp_type + "\n")
        sys.exit(1)
    experiment = ExperimentClass(args, target_dir = target_dir)


    if experiment.run():
        _logger.info("########## SUCCESS ##########")

if __name__ == "__main__":
    run()
