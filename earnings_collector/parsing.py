"""HTML extraction and SEC filing index parsing without third-party dependencies."""
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.suppressed = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'ix:hidden', 'head'}:
            self.suppressed.append(tag)
        if not self.suppressed:
            if tag in {'p', 'div', 'br', 'tr', 'h1', 'h2', 'h3', 'li'}:
                self.parts.append('\n')
            elif tag in {'td', 'th'}:
                self.parts.append('\t')

    def handle_endtag(self, tag):
        if self.suppressed:
            if tag == self.suppressed[-1]:
                self.suppressed.pop()
        elif tag in {'p', 'div', 'tr', 'h1', 'h2', 'h3', 'li'}:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)


def extract_text(raw):
    decoded = raw.decode('utf-8-sig', errors='replace')
    parser = TextParser()
    parser.feed(decoded)
    lines = [re.sub(r'[^\S\t\n]+', ' ', line).strip() for line in ''.join(parser.parts).splitlines()]
    return '\n'.join(line for line in lines if line)


class IndexParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'tr':
            self.row = []
        elif tag == 'td' and self.row is not None:
            self.cell = {'text': '', 'href': None}
        elif tag == 'a' and self.cell is not None:
            self.cell['href'] = attrs.get('href')

    def handle_data(self, data):
        if self.cell is not None:
            self.cell['text'] += data

    def handle_endtag(self, tag):
        if tag == 'td' and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def exhibits(raw, base_url):
    parser = IndexParser()
    parser.feed(raw.decode('utf-8', errors='replace'))
    results = []
    for row in parser.rows:
        if len(row) < 4 or not row[3]['text'].strip().upper().startswith('EX-99'):
            continue
        href = row[2]['href']
        if not href:
            continue
        url = urljoin(base_url, href)
        # Only accept exhibits inside this filing's archive directory.
        if not url.startswith(base_url) or urlparse(url).hostname != 'www.sec.gov':
            continue
        results.append({'url': url, 'description': row[1]['text'].strip(), 'type': row[3]['text'].strip()})
    return results


def is_earnings_release(text):
    """Conservative candidate classification, not a claim of verified financial data."""
    lower = text.lower()
    announcement = re.search(r'(reports?|announces?|announced)\b.{0,140}\b(results|earnings)', lower, re.S)
    period = re.search(r'quarter|fiscal|year ended|year-end|annual results', lower)
    financial = re.search(r'revenue|net income|earnings per share|net sales', lower)
    return bool(announcement and period and financial)
