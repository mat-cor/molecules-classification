"""
Class pubChemAPI for accessing and retrieving molecules information from pubChem
"""
import requests
from html.parser import HTMLParser
from xml.dom import minidom

import pandas as pd


def e_search(db, query):
    url2fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db="\
                + db + "&usehistory=y&term=" + query
    raw = requests.get(url2fetch).text

    kq = QueryKeyParser()
    kq.feed(raw)
    query_key = kq.q

    we = WebEnvParser()
    we.feed(raw)
    web_env = we.w

    return query_key, web_env


def get_summary(query_key, web_env, db):
    url2fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=" + db + "&version=2.0&query_key="\
                + query_key + "&WebEnv=" + web_env
    raw = requests.get(url2fetch).text
    return raw


def parse_summary(raw):
    ndata = raw.replace('\t', '').replace('\n', '')
    xmldoc = minidom.parseString(ndata)
    items = xmldoc.getElementsByTagName('DocumentSummary')
    cids, smiles, names, formulas = [], [], [], []
    for i in items:
        """
        First of all, verify if the compound has a MeSH Pharma annotation, otherwise it's useless to retrieve it
        """
        pharm = i.getElementsByTagName('PharmActionList')
        for p in pharm:
            """ If the node PharmActionList has children it means that the compound has pharma annotation"""
            if p.childNodes:
                for cnode in (i.getElementsByTagName('CID')):
                    if cnode.childNodes:
                        cids.append(str(cnode.childNodes[0].data))
                    else:
                        cids.append("?")
                for snode in (i.getElementsByTagName('CanonicalSmiles')):
                    if snode.childNodes:
                        smiles.append(str(snode.childNodes[0].data))
                    else:
                        smiles.append("?")
                for inode in (i.getElementsByTagName('MeSHHeadingList')):
                    if inode.childNodes:
                        names.append(str(inode.childNodes[0].childNodes[0].data))
                    else:
                        names.append("?")
                for fnode in (i.getElementsByTagName('MolecularFormula')):
                    if fnode.childNodes:
                        formulas.append(str(fnode.childNodes[0].data))
                    else:
                        formulas.append("?")

    return cids, smiles, names, formulas


def mesh_link_parser(raw):
    ndata = raw.replace('\t', '').replace('\n', '')
    xmldoc = minidom.parseString(ndata)
    items = xmldoc.getElementsByTagName('Link')
    meshids = []
    for i in items:
        ids = i.getElementsByTagName('Id')
        for uid in ids:
            meshids.append(str(uid.childNodes[0].data))
    return meshids


def get_mesh_summary(meshid):
    meshidstr = ",".join(meshid)

    url2fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=mesh&id=" + meshidstr
    raw = requests.get(url2fetch).text

    return raw


def get_mesh_info(raw):
    ndata = raw.replace('\t', '').replace('\n', '')
    xmldoc = minidom.parseString(ndata)
    items = xmldoc.getElementsByTagName('Item')
    treeNumbers = []
    meshTerms = []
    for i in items:
        if i.attributes['Name'].value == 'TreeNum':
            treeNumbers.append(str(i.childNodes[0].data))
        if i.attributes['Name'].value == 'DS_MeshTerms':
            meshTerms.append(str(i.childNodes[0].childNodes[0].data))

    return treeNumbers, meshTerms


def get_mesh_id(cid):
    url2fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pccompound&version=2.0" \
                "&db=mesh&id=" + cid + "&linkname=pccompound_mesh_pharm"
    raw = requests.get(url2fetch).text

    return mesh_link_parser(raw)


def get_mesh_info_from_cid(cid):
    meshidlist = get_mesh_id(cid)
    raw = get_mesh_summary(meshidlist)
    return get_mesh_info(raw)


def get_data_frame(raw):

    cids, smiles, names, formulas = parse_summary(raw)
    treeids, terms = [], []

    for c in cids:
        tid, ts = get_mesh_info_from_cid(c)
        treeids.append(tid)
        terms.append(ts)

    d = {'CID': cids, 'SMILES': smiles, 'Name': names, 'Formula': formulas, 'Terms': terms, 'TreeIds': treeids}
    df = pd.DataFrame(data=d)
    df = df.sort_values('CID')
    return df.reset_index(drop=True)


class QueryKeyParser(HTMLParser):
    def reset(self):
        self.nextq = False
        self.q = ''
        HTMLParser.reset(self)

    def handle_starttag(self, tag, attrs):
        # print(tag, " ", attrs)
        if tag == "querykey":
            self.nextq = True

    def handle_data(self, data):
        if self.nextq:
            self.q = data
            self.nextq = False


class WebEnvParser(HTMLParser):
    def reset(self):
        self.nextw = False
        self.w = ''
        HTMLParser.reset(self)

    def handle_starttag(self, tag, attrs):
        # print(tag, " ", attrs)
        if tag == "webenv":
            self.nextw = True

    def handle_data(self, data):
        if self.nextw:
            self.w = data
            self.nextw = False
