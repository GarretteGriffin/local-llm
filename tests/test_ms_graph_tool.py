from tools.ms_graph import extract_microsoft_urls, share_id_from_url


def test_extract_microsoft_urls_filters_hosts():
    text = (
        "See https://example.com/a and "
        "https://contoso.sharepoint.com/sites/x/Shared%20Documents/test.docx plus "
        "https://1drv.ms/u/s!abc123"
    )
    urls = extract_microsoft_urls(text)
    assert "https://contoso.sharepoint.com/sites/x/Shared%20Documents/test.docx" in urls
    assert "https://1drv.ms/u/s!abc123" in urls
    assert "https://example.com/a" not in urls


def test_share_id_from_url_is_prefixed_and_unpadded():
    url = "https://contoso.sharepoint.com/:w:/s/site/EaBcDeFgHiJ?e=abcd"
    sid = share_id_from_url(url)
    assert sid.startswith("u!")
    assert "=" not in sid
