from pyDataverse.api import NativeApi


def upload_proc_be(config):
  """ Uploads processed building electricity data."""

  # get base url and api token
  base_url = config['dataverse']['base_url']
  with open(config['dataverse']['path_to_token'], 'r') as file:
    api_token = file.read()

  api = NativeApi(base_url, api_token)

  # do some test
  resp = api.get_info_version()
  resp.json()


