from app import routes
from flask import Blueprint, render_template, request
from app.publication import perform_search
from math import ceil
import time
from app.clustering import DocumentClustering

bp = Blueprint('bp', __name__)
clusterer = DocumentClustering()


def paginate_results(results, page, per_page):
    total_results = len(results)
    total_pages = ceil(total_results / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = results[start:end]
    return paginated_results, total_pages, total_results


@bp.route('/')
def index():
    return render_template('index.html', title='Coventry University School of Computing, Mathematics and Data Sciences Research Publications Research Publications Search Engine')


@bp.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    page = request.args.get('page', 1, type=int)
    per_page = 10
    start_time = time.time()
    results = perform_search(query)
    end_time = time.time()
    search_time = round(end_time - start_time, 2)
    paginated_results, total_pages, total_results = paginate_results(
        results, page, per_page)
    return render_template('search_results.html', title='Search Results', query=query, results=paginated_results, page=page, total_pages=total_pages, total_results=total_results, search_time=search_time)


@bp.route('/cluster')
def cluster_index():
    return render_template('cluster.html', title='Document Clustering')


@bp.route('/cluster_result', methods=['POST'])
def cluster_document():
    print("sdf")
    data = request.form['document']
    result = clusterer.cluster_document(data)
    return result
