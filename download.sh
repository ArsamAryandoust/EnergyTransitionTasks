export API_TOKEN="$(cat ../.aa_secrets/dataverse_token)"
export SERVER_URL=https://dataverse.harvard.edu
export PERSISTENT_ID=doi:10.7910/DVN/H26M0K

# download file
curl -L -O -J -H "X-Dataverse-key:$API_TOKEN" $SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID

# create a "data" directory if doesn't exist already
mkdir data

# unzip downloaded file to data directory
unzip -q dataverse_files.zip -d data/
rm dataverse_files.zip

# extract the .tar.gz files from unzipped directory
tar -xf data/'Building Electricity.tar.gz' -C data/
rm data/'Building Electricity.tar.gz'


