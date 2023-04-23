import pandas as pd
import json
import gc

from load_config import config_PA


def process_all_datasets(config: dict, save: bool):
  """
  Process all datasets for Polianna task.
  """
  
  # iterate over all subtasks
  for subtask in config['Polianna']['subtask_list']:
    
    # show whats going on
    print("Processing Polianna dataset {}.".format(subtask))
    
    # augment config with current subtask data
    config_polianna = config_PA(config, subtask, save)
    
    # import all data
    df_data, df_meta = import_all_data(config_polianna)

    # clean data
    df_data = clean_data(df_data)
    
    # merge selected information from meta data and merge into cleaned data df    
    df_data = merge_data(config_polianna, df_data, df_meta)
    
    # free up memory
    del df_meta
    gc.collect()

    # create data points
    df_data = expand_data(df_data)
    
    # ordinally encode categorical features
    df_data, _ = encode_features(config_polianna, df_data, save)
    
    # encode main article text
    df_data, _, _ = encode_articles(config_polianna, df_data, save)
    
    # encode labels
    df_data = encode_labels(config_polianna, df_data, save)
    
    # split train, val, test
    split_train_val_test(config_polianna, df_data, save)
    
    # create coding scheme dictionary
    _ = create_and_save_handmade_coding(config_polianna, save)


def encode_labels(config_polianna: dict, df_data: pd.DataFrame, save: bool
  ) -> (pd.DataFrame):
  """
  """
  
  # create empty dict to save all annotation labels
  label_dict = {}
  label_set = set()
  label_tag_list_dict = {}

  ### Gather detailed annotation information

  # iterate over all annotation data points
  for index_data, annotation_point in enumerate(df_data['annotation']):
      
    # split into single annotations
    annotation_list = annotation_point.split(',')

    # create empty annotation dict and tag list for saving labels
    anno_dict = {}
    tag_list = []
    
    # iterate over each annotation
    for index_anno, anno in enumerate(annotation_list):
        
      # split single annotation again
      anno_split = anno.split()
      
      # iterate over each part of split annotation list
      for anno_part in anno_split:
          
          # get start, stop and tag only for labels
          if 'start:' in anno_part:
            start = anno_part[6:]
          elif 'stop:'in anno_part:
            stop = anno_part[5:]
          elif 'tag:' in anno_part:
            tag = anno_part[4:]
      
      # set annotation dictionary
      anno_dict[index_anno] = {
        'start' : start,
        'stop' : stop,
        'tag' : tag
      }
      
      # add to tags list
      tag_list.append(tag)
        
    # save records
    label_set = label_set.union(set(tag_list)) 
    label_dict[index_data+1] = anno_dict
    label_tag_list_dict[index_data+1] = tag_list
      
  # drop annotation column
  df_data.drop(columns=['annotation'], inplace=True)

  # transform set to list
  label_set = list(label_set)

  # sort list
  label_set.sort()

  # create a label encoding
  label_enc_dict = {}
  for index_elem, label in enumerate(label_set):
    label_enc_dict[label] = index_elem
      

  # if subtask is article level, create histogram over labels
  if config_polianna['subtask'] == 'article_level':
    # set empty dictionary to save labels
    labels = {}

    # iterate over all label list elements
    for key_data, value_data in label_tag_list_dict.items():

      # label distribution dict
      label_dist_dict = {}

      # iterate over all possible labels
      for key_enc, value_enc in label_enc_dict.items():

        # count number of labels in list
        n_labels = value_data.count(key_enc)

        # save 
        label_dist_dict[key_enc] = n_labels

      # save labels record
      labels[key_data] = label_dist_dict
        
    # create a dataframe from dictionary
    df_labels = pd.DataFrame.from_dict(labels, orient='index').reset_index(drop=True)
    
    # concatenate into features dictionary
    df_data = pd.concat([df_data, df_labels], axis=1)
        
  # if subtask is text level, keep detailed annotations as labels
  elif config_polianna['subtask'] == 'text_level':
      
    # transform tag entries into corresponding encodings
    for key, value in label_dict.items():
      for key_1, value_1 in value.items():
        value_1['tag'] = label_enc_dict[value_1['tag']]
    
    # set labels to transformed label dict
    labels = label_dict
    
    # save labels
    if save:
      
      # set saving paths
      saving_path_annotation = (
        config_polianna['path_to_data_subtask_add'] + 'annotation_labels.json')
      saving_path_encoding = (
        config_polianna['path_to_data_subtask_add'] + 'encoding_labels.json')
      
      # save file
      with open(saving_path_annotation, "w") as saving_file:
        json.dump(labels, saving_file) 
      with open(saving_path_encoding, "w") as saving_file:
        json.dump(label_enc_dict, saving_file) 

  
  return df_data

def encode_articles(config_polianna: dict, df_data: pd.DataFrame, save: bool
  ) -> (pd.DataFrame, dict, dict):
  """
  """
  
  # set empty dictionary for recording encoding scheme and saving as json
  art_enc_dict_text = {}
  art_enc_dict_token = {}

  # iterate over every data point's article
  for art_index, (article_text, article_token) in enumerate(
    zip(df_data['article_text'], df_data['article_token'])):
      
    # split the tokenized article content by comma into list
    article = article_token.split(',')
    
    # remove all entries containing 'token'
    article = [item for item in article if not 'token' in item]
    
    # remove two sets of irregular entries by following two conditions
    article = [item for item in article if 'start' in item.split()[0]]
    article = [item for item in article if not 'text' in item.split()[-1]]
    
    # set empty token dict to fill for each article
    token_dict = {}
    
    # iterate over all tokens in article
    for token_index, token in enumerate(article):
        
      # split token into start, stop, text and tag_count elements
      token_split = token.split()
      
      # save token start, stop and text
      token_dict[token_index] = {
        'start' : token_split[0][6:],
        'stop' : token_split[1][5:],
        'text' : token_split[2][5:]
      }
    
    art_enc_dict_text[art_index+1] = article_text
    art_enc_dict_token[art_index+1] = token_dict
  
  # save
  if save:
  
    # set saving path
    saving_path_text = (
      config_polianna['path_to_data_subtask_add'] + 'article_text.json')
    saving_path_token = (
      config_polianna['path_to_data_subtask_add'] + 'article_tokenized.json')
    
    # save file
    with open(saving_path_text, "w") as saving_file:
      json.dump(art_enc_dict_text, saving_file) 
    with open(saving_path_token, "w") as saving_file:
      json.dump(art_enc_dict_token, saving_file) 

  
  # drop old columns
  df_data.drop(columns=['article_text', 'article_token'], inplace=True)

  # create new column
  df_data['article'] = range(1, len(df_data) + 1)

  return df_data, art_enc_dict_text, art_enc_dict_token


def encode_features(config_polianna: dict, df_data: pd.DataFrame, save: bool
  ) -> (pd.DataFrame, dict):
  """
  """
  
  # set list of columns we want to ordinally encode
  enc_cols = ['form', 'treaty']

  # set empty dictionary for recording encoding scheme and saving as json
  feat_enc_dict = {}

  # iterate over all requested columns
  for col in enc_cols:
      
    # get set of values in column
    enc_set = set(df_data[col])
    
    # set empty encoding dictionary to save values
    enc_dict = {}
    
    # iterate over set to be encoded
    for entry_index, entry_set in enumerate(enc_set):
        enc_dict[entry_set] = entry_index
    
    # transform dataframe
    df_data = df_data.replace({col: enc_dict})
    
    # save encoding scheme
    feat_enc_dict[col] = enc_dict
  
  # save results
  if save:
  
    # set saving path
    saving_path = (
      config_polianna['path_to_data_meta'] + 'encoding_features.json')
    
    # save file
    with open(saving_path, "w") as saving_file:
      json.dump(feat_enc_dict, saving_file) 
  
  return df_data, feat_enc_dict


def expand_data(df_data: pd.DataFrame) -> (pd.DataFrame): 
  """
  """
  
  ### expand time features ###

  # split time stamp
  df_data[['year', 'month', 'day']] = df_data.date.str.split("-", expand = True)

  # get columns
  cols = df_data.columns.to_list()

  # remove date and rearrange others to be first
  cols.remove('date'), cols.remove('year')
  cols.remove('month'), cols.remove('day')
  cols = ['year', 'month', 'day'] + cols

  # set new dataframe
  df_data = df_data[cols]
  
    
  ### rearrange 'form' ###

  # get columns
  cols = df_data.columns.to_list()

  # remove 'form'
  cols.remove('form')

  # insert at beginning
  cols.insert(3, 'form')
  
  # set new dataframe
  df_data = df_data[cols]
    
  return df_data


def split_train_val_test(config_polianna: dict, df_data: pd.DataFrame, 
  save: bool):
  """
  """
  
  pass
  
  
def clean_data(df_data: pd.DataFrame) -> (pd.DataFrame):
  """
  """
  
  # get the indices where curation is missing
  index_list_miss = df_data[df_data['Curation'] == '[]'].index
  
  # drop rows by index
  df_data.drop(index=index_list_miss, inplace=True)
  
  return df_data


def merge_data(config_polianna: dict, df_data: pd.DataFrame, 
  df_meta: pd.DataFrame) -> (pd.DataFrame):
  """
  """
  
  ### merge into df_data ###

  # transform filename column to match df_meta
  df_data.rename(columns={'Unnamed: 0': 'Filename'}, inplace=True)

  # shorten filename length to match df_meta
  df_data['Filename'] = df_data['Filename'].apply(
    lambda x: x[:len('EU_32009B0632')]
  )

  # merge
  df_data = df_data.merge(df_meta, on='Filename')
  
  # free up memory
  del df_meta
  _ = gc.collect()
  
  
  ### set chosen columns ###
  
  # set new columns
  df_data = df_data[config_polianna['data_col_list'] ]
    
  # rename columns
  df_data = df_data.rename(columns=config_polianna['rename_col_dict'])
  
  
  ### do some cleaning
  
  # drop rows where entry is missing
  df_data.dropna(inplace=True, ignore_index=True)
  
  return df_data


def import_all_data(config_polianna: dict) -> (pd.DataFrame, pd.DataFrame):
  """
  """
  
  # set paths to data
  path_to_meta_csv = (
    config_polianna['path_to_data_raw_meta'] + 'EU_metadata.csv')
  path_to_data_csv = (
    config_polianna['path_to_data_raw_dataframe']+'preprocessed_dataframe.csv')
  
  # load dataframes from csv
  df_meta = pd.read_csv(path_to_meta_csv)
  df_data = pd.read_csv(path_to_data_csv)
  
  return df_data, df_meta
  

def create_and_save_handmade_coding(config_polianna: dict, save: bool
  ) -> (dict):
  """
  """
  
  # manually set coding scheme as dictionary
  coding_dict = {
    "Instrumenttypes" : {
      "InstrumentType" : {
        "Edu_Outreach":"Education and outreach:\r\nPolicies designed to increase knowledge, awareness, and training among relevant stakeholders or users, including information campaigns, training programs, labelling schemes",
        "FrameworkPolicy":"Framework policy:\r\nRefers to the processes undertaken to develop and implement policies. This generally covers strategic planning documents and strategies that guide policy development. It can also include the creation of specific bodies to further policy aims, making strategic modifications, or developing specific programs.",
        "PublicInvt":"Public Investment:\r\nPolicies guiding investment by public bodies. These include government procurement programs (e.g. requirement to purchase LIB-powered electric vehicles) and infrastructure investment (e.g. charging infrastructure).",
        "RD_D":"Research, Development & Demonstration (RD&D):\r\nPolicies and measures for the government to invest directly in or facilitate investment in technology research, development, demonstration and deployment activities.",
        "RegulatoryInstr":"Regulatory Instrument:\r\nCovers a wide range of instruments by which a government will oblige actors to undertake specific measures and/or report on specific information. Examples include obligations on companies to reduce energy consumption, produce or purchase a certain amount of LIB-powered electric vehicles or requirements to report on GHG emissions or energy use.",
        "Subsidies_Incentives":"Subsidies and direct incentives:\r\nPolicies to stimulate certain activities, behaviours or investments through subsidies or rebates for, e.g.,  the purchase of LIB-powered electric vehicles, grants, preferential loans and third-party financing for LIB manufacturing.",
        "TaxIncentives":"Tax incentives:\r\nPolicies to encourage or stimulate certain activities or behaviours through tax exemptions, tax reductions or tax credits on the purchase or installation of certain goods and services.",
        "TradablePermit":"Tradable Permit:\r\nRefers to GHG emissions trading schemes or white certificate systems related to energy efficiency or energy savings obligations. In the former, industries must hold permits to cover their GHG emissions; if they emit more than the amount of permits they hold, they must purchase permits to make up the shortfall, creating an incentive to reduce energy use. White certificate schemes create certificates for a certain quantity of energy saved (for example, one MWh); regulated entities must submit enough certificates to show they have met energy saving obligations. If they are short, this must be made-up through measures that reduce energy use, or through purchase of certificates.",
        "Unspecified":"General descriptions of instrument types without specifying which on (\"mechanism\", \"measure\", etc.).",
        "VoluntaryAgrmt":"Voluntary Agreement:\r\nRefers to measures that are undertaking voluntarily by government agencies or industry bodies, based on a formalized agreement. There are incentives and benefits to undertaking the action, but generally few legal penalties in case of non-compliance. The scope of the action tends to be agreed upon in concert with the relevant actors; for example, agreements to report LIB-related RD&D activities or nonbinding commitments to cooperation between actors."
      }
    },
    "Policydesigncharacteristics": {
      "Objective": {
        "Objective_QualIntention":"Qualitative intention: \r\nA qualitatively stated intention or objective of the policy. This lacks a specific quantity that is targeted, for example increasing the amount of hydrogen produced with renewable electricity sources. Also includes references to unspecified targets.",
        "Objective_QualIntention_noCCM":"Qualitative intention not mitigation: \r\nA qualitatively stated intention or objective of the policy, not pertaining to climate change mitigation (e.g. jobs). This lacks a specific quantity that is targeted.",
        "Objective_QuantTarget":"Quantitative target: \r\nA quantitative target or objective of the policy.",
        "Objective_QuantTarget_noCCM":"Quantitative target not mitigation: \r\nA quantitative target or objective of the policy, not pertaining to climate change mitigation (e.g. jobs)."
      },
      "Reference": {
        "Ref_OtherPolicy":"Reference to other policy: \r\nExternal legislative text referenced for objectives, definitions, constraints, or for other reasons. No amendments.",
        "Ref_PolicyAmended":"Amendment of policy: \r\nAmendment of another policy, or repeal thereof, that is made through this legislation.",
        "Ref_Strategy_Agreement":"Reference to strategy or agreement: \r\nReference to treaties, constitutions, agreements, white papers, blue prints, overarching strategies. For example the Paris Agreement"
      },
      "Actor": {
        "Addressee_default":"Default addressee: \r\nThe individual or entity that the rule applies to and needs to ensure its implementation.",
        "Addressee_monitored":"Monitored addressee: \r\nAn individual or entity monitored for the outcome of the policy, through report, review, or audit. Formerly known as actor monitored.",
        "Addressee_resource":"Resources addressee [Addressee_resource]: The actor that receives a resource. Formerly resources_recipient.",
        "Addressee_sector":"Sector addressee: \r\nRelevant sectors that are covered by the policy. Formerly scope.",
        "Authority_default":"Default authority: \r\nThe individual or entity that is making the rule, ensuring its implementation, for enforcing the rule and may apply sanctioning, including an existing individual or entity empowered, directed or required to implement. Formerly known as enforcement_actor but more comprehensive.",
        "Authority_established":"Newly established authority: \r\nA newly established entity that is ensuring the policy\u2019s implementation.",
        "Authority_legislative":"Legislative authority: \r\nThe individual or entity that is drafting or voting on legislation.",
        "Authority_monitoring":"Monitoring authority: \r\nAn individual or entity responsible for monitoring the outcome of the policy, through report, review, or audit. All entities that are part of the monitoring process, and not the primary monitored entity."
      },
      "Resource" : {
        "Resource_MonRevenues":"Monetary_revenues: \r\nProvisions that affect revenue (positively or negatively). Can be a concrete sum or unspecific assumption such as \u201cincrease revenue\u201d. This includes, e.g., tax credits (negative), tolls, fees, customs (positive).",
        "Resource_MonSpending":"Monetary_spending: \r\nResources that are provided through spending. Can be a concrete sum or unspecific assumption such as \u201cmore spending on\u2026\u201d. This includes grants, subsidies, allocations of funds.",
        "Resource_Other":"Other resource type: \r\nOther resources such as personnel, facilities/equipment, or emissions allowances."
      },
      "Time" : {
        "Time_Compliance":"Compliance time: \r\nThe deadline or time frame for compliance with the regulation.",
        "Time_InEffect":"In effect time: The start date or effective date of the policy.",
        "Time_Monitoring":"Monitoring time: Deadlines, time frames and frequencies related to monitoring.",
        "Time_PolDuration":"Policy duration time:\r\nA set time frame that the policy is in place or a deadline by when it expires.",
        "Time_Resources":"Resources time: \r\nTemporal provisions around the resource allocation."
      },
      "Compliance" : {
        "Form_monitoring":"Monitoring form: \r\nThe form of the monitoring (provisions relating to report, review, or audit; standards and certification schemes)",
        "Form_sanctioning":"Sanctioning form: \r\nSanctioning provisions and measures."
      },
      "Reversibility" : {
        "Reversibility_policy":"Pol_duration_provision: \r\nA provision for the extension or termination of the policy."
      }
    },
    "Technologyandapplicationspecificity": {
      "EnergySpecificity": {
        "Energy_LowCarbon":"Low-carbon energy source or carrier: A low-carbon energy source or energy carrier (includes biomass and nuclear).",
        "Energy_Other":"Other energy source or carrier: Other energy source or energy carrier (includes fossil fuels)."
      },
      "ApplicationSpecificity" : {
        "App_LowCarbon":"Low-carbon application: \r\nApplication of a low-carbon technology or low-carbon application of a technology.",
        "App_Other":"Other application: \r\nOther application with no direct role for decarbonization."
      },
      "TechnologySpecificity" : {
        "Tech_LowCarbon":"Low-carbon technology: \r\nLow-carbon technology, including renewable energy generation, storage, efficiency at various levels of precision.",
        "Tech_Other":"Other technology: \r\nOther technologies with no direct role for decarbonization."
      }
    }
  }
  
  # save only if chosen so
  if save:
    
    # set saving path
    saving_path = (
      config_polianna['path_to_data_meta'] + 'coding_scheme.json')
    
    # save file
    with open(saving_path, "w") as saving_file:
      json.dump(coding_dict, saving_file) 
  
  return coding_dict

