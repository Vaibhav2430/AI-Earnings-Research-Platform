import argparse
import os
import sys
from pathlib import Path

from .collector import collect
from .sec import CollectionError, SecClient


def main():
    parser = argparse.ArgumentParser(description='Automatically collect SEC earnings materials for a stock ticker.')
    parser.add_argument('ticker', help='Stock ticker, for example AAPL')
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--user-agent', default=os.environ.get('SEC_USER_AGENT'), help='Application name and contact email; defaults to SEC_USER_AGENT')
    parser.add_argument('--refresh', action='store_true', help='Download archived documents again instead of using the cache')
    parser.add_argument('--max-earnings-filings', type=int, default=8)
    args = parser.parse_args()
    if not args.user_agent or '@' not in args.user_agent or '\n' in args.user_agent or '\r' in args.user_agent:
        parser.error('Set SEC_USER_AGENT to an application name and real contact email, or supply --user-agent.')
    if not 1 <= args.max_earnings_filings <= 50:
        parser.error('--max-earnings-filings must be between 1 and 50')
    try:
        client = SecClient(args.user_agent, args.output / '.cache', args.refresh)
        path, result = collect(args.ticker, client, args.output, args.max_earnings_filings)
    except (CollectionError, OSError) as exc:
        print(f'Collection failed: {exc}', file=sys.stderr)
        return 1
    print(f"{result['company_name']} ({result['ticker']}): {result['status']}")
    print(f"Saved {len(result['documents'])} documents. Manifest: {path.resolve()}")
    for warning in result['warnings']:
        print(f'Note: {warning}')
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    sys.exit(main())
