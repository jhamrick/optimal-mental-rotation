#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ConfigParser import SafeConfigParser
from path import path
import logging
import urllib2

logger = logging.getLogger('experiment.fetch_data')


def add_auth(url, username, password):
    """Add HTTP authencation for opening urls with urllib2.

    Based on http://www.voidspace.org.uk/python/articles/authentication.shtml

    """

    # this creates a password manager
    passman = urllib2.HTTPPasswordMgrWithDefaultRealm()

    # because we have put None at the start it will always use this
    # username/password combination for urls for which `theurl` is a
    # super-url
    passman.add_password(None, url, username, password)

    # create the AuthHandler
    authhandler = urllib2.HTTPBasicAuthHandler(passman)

    # All calls to urllib2.urlopen will now use our handler Make sure
    # not to include the protocol in with the URL, or
    # HTTPPasswordMgrWithDefaultRealm will be very confused.  You must
    # (of course) use it when fetching the page though.
    opener = urllib2.build_opener(authhandler)
    urllib2.install_opener(opener)


def fetch(site_root, filename, version, force=False):
    """Download `filename` from `site_root` and save it in the
    human-raw/`experiment` data folder.

    """

    # get the url
    url = path(site_root).joinpath("%s_all.csv" % filename)

    # get the destination to save the data, and don't do anything if
    # it exists already
    DATA_PATH = path(config.get("paths", "data"))
    dest = DATA_PATH.joinpath("human-raw", version, url.name)
    if dest.exists() and not force:
        return

    # try to open it
    try:
        handler = urllib2.urlopen(url)
    except IOError as err:
        if getattr(err, 'code', None) == 401:
            logger.error("Server authentication failed.")
            raise err
        else:
            raise

    # download the data
    data = handler.read()
    logger.info("Fetched succesfully: %s", url)

    # make the destination folder if it doesn't exist
    if not dest.dirname().exists():
        dest.dirname().makedirs_p()

    # write out the data file
    with open(dest, "w") as fh:
        fh.write(data)
    logger.info("Saved to '%s'", dest.relpath())


if __name__ == "__main__":

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-c", "--config",
        default="config.ini",
        help="path to configuration file")
    parser.add_argument(
        "-u", "--user",
        default=None,
        help="Username to authenticate to the server.")
    parser.add_argument(
        "-p", "--password",
        default=None,
        help="Password to authenticate to the server.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Overwrite existing data.")

    args = parser.parse_args()

    # load configuration
    config = SafeConfigParser()
    config.read(args.config)

    loglevel = config.get("global", "loglevel")
    logging.basicConfig(level=loglevel)

    version = config.get("global", "version")
    address = config.get("experiment", "fetch_path")

    # prompt for the username if it wasn't given
    if args.user is None:
        username = raw_input("Username: ")
    else:
        username = args.user

    # prompt for the password if it wasn't given
    if args.password is None:
        password = raw_input("Password: ")
    else:
        password = args.password

    # create the authentication handler to the server
    add_auth(address, username, password)

    # fetch and save the data files
    files = ["trialdata", "eventdata", "questiondata"]
    for filename in files:
        fetch(address, filename, version, args.force)
