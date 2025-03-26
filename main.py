import argparse

import utils.logging_utils as log_utils

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train hybrid trading strategy models with uncertainty quantification')
    
    parser.add_argument('--broker-func', action='store_true', default=True, help='Test all broker functinoality')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    if args.broker_func:
        logger = log_utils.setup_logging(log_to_file=True)
        logger.info("Trading system starting...")
        print(hei)

if __name__ == "__main__":
    exit(main())