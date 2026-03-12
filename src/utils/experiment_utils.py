
import argparse


def parse_terminal_arguments():
    """ Reads experiment arguments """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument( "-c", "--config_file", action="store", type=str, required=True,
                                 help="JSON file with experiment parameters.")
    argument_parser.add_argument("-i", "--run_index", action="store", type=int, default=0,
                                 help="This determines the random seed for the experiment.")
    argument_parser.add_argument("-v", "--verbose", action="store_true", default=False)
    argument_parser.add_argument("--gpu_index", action="store", type=int, default=0)
    argument_parser.add_argument("--tensorboard", action="store_true", default=False,
                                 help="Enable TensorBoard logging for this run.")
    argument_parser.add_argument("--tensorboard_log_dir", action="store", type=str, default=None,
                                 help="Base directory where TensorBoard logs are written.")
    return argument_parser.parse_args()


def parse_plots_and_analysis_terminal_arguments():
    """ Reads analysis arguments """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-c", "--config_file", action="store", type=str, required=True,
                                 help="JSON file with analysis configurations.")
    argument_parser.add_argument("-s", "--save_plot", action="store_true", default=False)
    argument_parser.add_argument("--debug", action="store_true", default=False)
    return argument_parser.parse_args()
