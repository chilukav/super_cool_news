import os
import time
import json
import hashlib
import logging
import argparse
import collections
import datetime as dt

import feedparser


MAIN_LOOP_PERIOD_SEC = 60 * 60
logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


def parse_args():
    parser = argparse.ArgumentParser("RSS feed parser")
    parser.add_argument(
        "path",
        help="Path to base directory of sources."
             " Base directory should contain `sources.txt` file with list of RSS links."
             "Links should be specified on separate lines with topic set after colon\n"
             "For instance: https://rss.feed.address.com:Sports\n"
             "Entries of each feed will appear at <base directory>/YYYY-mm-dd/<topic>-<title> text files. "
             "Each entry represented in this files as title and entry text. "
             "Entries separate from each other with 3 new lines"
    )
    return parser.parse_args()


class FeedFetcher:
    def __init__(self, base_dir):
        self.base_dir = os.path.abspath(base_dir)
        self.sources = collections.defaultdict(list)
        self.journal = {}
        self.journal_path = os.path.join(self.base_dir, ".fetch.jrnl")
        self.load_journal()
        self.load_sources()

    @property
    def source_file(self):
        ret = os.path.join(self.base_dir, "sources.txt")
        if not os.path.exists(ret) or not os.path.isfile(ret):
            raise RuntimeError(f"No sources.txt found at `{self.base_dir}`")
        return ret

    def load_sources(self):
        with open(self.source_file) as fd:
            for line in fd:
                link, topic = line.rsplit(":", maxsplit=1)
                self.sources[topic].append(link)
        logger.info("Source list successfully loaded, %s topics found", len(self.sources))

    def dump_journal(self):
        with open(self.journal_path, "w") as fd:
            json.dump(self.journal, fd)

    def load_journal(self):
        if not os.path.exists(self.journal_path):
            logger.debug("No journal file at %s found, leaving journal empty", self.journal_path)
            return

        with open(self.journal_path) as fd:
            self.journal = json.load(fd)

    def download(self, link, topic):
        logger.info("Fetching feed at %s - %s", link, topic)
        feed = feedparser.parse(link)
        entries = feed.get("entries", [])
        if not entries:
            logger.error("No entries at %s - %s", link, topic)
            return

        feed_hash = hashlib.md5(bytes(json.dumps(entries, ensure_ascii=False), "utf-8")).hexdigest()
        cached = self.journal.get(link, {})
        cached_hash = cached.get("hash")
        if cached_hash and feed_hash == cached_hash:
            cached_dt = cached.get("date")
            logger.warning("Feed at `%s` not changed since %s", link, cached_dt)
            return

        title = feed["feed"]["title"]
        today = dt.datetime.today()
        link_dir = os.path.join(self.base_dir, str(today.date()), f"{topic}-{title}")
        os.makedirs(os.path.dirname(link_dir), exist_ok=True)
        with open(link_dir, "a+") as fd:
            for entry in feed["entries"]:
                fd.write(entry["title"])
                fd.write("\n")
                fd.write(entry["summary"])
                fd.write("\n" * 3)

        self.journal[link] = {"hash": feed_hash, "date": today.isoformat()}
        logger.info("Done parsing feed %s - %s", link, topic)

    def run(self):
        logger.info("Starting to fetch sources")
        for topic, links in self.sources.items():
            for link in links:
                self.download(link, topic)
        logger.info("Done fetching sources")


def main():
    args = parse_args()
    fetcher = FeedFetcher(args.path)
    for iteration in range(1, 25):
        logger.info("Starting iteration #%s", iteration)
        fetcher.run()
        logger.info("Iteration #%s ended, going to sleep for a while", iteration)
        time.sleep(MAIN_LOOP_PERIOD_SEC)
    fetcher.dump_journal()
    logger.info("All done.")


if __name__ == "__main__":
    main()
