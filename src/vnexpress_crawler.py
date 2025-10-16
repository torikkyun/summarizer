import requests
from bs4 import BeautifulSoup
import json


def crawl_article_content(url):
    """Crawl nội dung chi tiết một bài viết"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        title_tag = soup.find("h1", class_="title-detail")
        title = title_tag.text.strip() if title_tag else ""

        sapo_tag = soup.find("p", class_="description")
        sapo = ""
        if sapo_tag:
            location_tag = sapo_tag.find("span", class_="location-stamp")
            if location_tag:
                location = location_tag.get_text(strip=True)
                location_tag.extract()
                sapo_text = sapo_tag.get_text(strip=True)
                sapo = f"{location} - {sapo_text}"
            else:
                sapo = sapo_tag.get_text(strip=True)

        content_div = soup.find("article", class_="fck_detail")
        if content_div:
            paragraphs = content_div.find_all("p", class_="Normal")
            content = " ".join([p.text.strip() for p in paragraphs])
        else:
            content = ""

        if title and sapo and content and len(content) > 200:
            return {
                "title": title,
                "text": content,
                "summary": sapo,
                "url": url,
            }
    except Exception as e:
        print(f"❌ Lỗi khi crawl {url}: {e}")
    return None


def crawl_vnexpress(num_articles=100):
    """Crawl dữ liệu từ VnExpress"""
    articles = []
    base_url = "https://vnexpress.net"
    categories = [
        "/tin-tuc-24h",
        "/kinh-doanh",
        "/giai-tri",
        "/the-thao",
        "/khoa-hoc",
    ]

    print(f"🕷️ Đang crawl {num_articles} bài viết từ VnExpress...")

    for category in categories:
        if len(articles) >= num_articles:
            break
        try:
            response = requests.get(base_url + category, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            article_links = soup.find_all("h3", class_="title-news")
            for link_tag in article_links[:50]:
                if len(articles) >= num_articles:
                    break
                link = link_tag.find("a")
                if not link:
                    continue
                article_url = link.get("href")
                if (
                    article_url
                    and isinstance(article_url, str)
                    and not article_url.startswith("http")
                ):
                    article_url = base_url + article_url
                article_data = crawl_article_content(article_url)
                if article_data:
                    articles.append(article_data)
                    print(f"✅ Đã crawl: {len(articles)}/{num_articles}")
        except Exception as e:
            print(f"❌ Lỗi khi crawl category {category}: {e}")
            continue
    return articles


def crawl_vietnamese_news(
    output_file="vietnamese_news_dataset.json", num_articles=100
):
    data = crawl_vnexpress(num_articles=num_articles)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Đã lưu {len(data)} bài viết vào {output_file}")
    return data
