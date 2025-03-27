import argparse
import utils.logging_utils as log_utils
from broker.capital_com.capitalcom import CapitalCom

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train hybrid trading strategy models with uncertainty quantification')
    
    parser.add_argument('--broker-func', action='store_true', default=True, help='Test all broker functionality')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger = log_utils.setup_logging(log_to_file=False)
    
    if args.broker_func:
        broker = CapitalCom()
        broker.start_session()
        # broker.session_details(print_answer=True)
        # broker.switch_active_account(print_answer=True)
        # broker.list_all_accounts(print_answer=True)
        # broker.get_historical_data(epic="SILVER", resolution="MINUTE_5",
        #                            from_date="2020-02-24T00:00:00", to_date="2020-02-24T01:00:00",
        #                            print_answer=True)
        broker.fetch_and_save_historical_prices(epic="GBPUSD", resolution="MINUTE_5",
                                                from_date="2024-01-01T00:00:00", to_date="2025-01-01T01:00:00",
                                                print_answer=False)
        
        broker.end_session()

if __name__ == "__main__":
    exit(main())