# TODO remove me

import datetime as dt
import json
from django.conf import settings
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import load_only

from scraper_common import db
from scraper_common import storage
from scraper_common.db import models
from scraper.clusterizer import config


class ClusteringRunner(storage.Storage):
    PARAMETERS = [
        {
            'code': 'cluster_method',
            'description': u'метод кластеризации',
            'value': 'kmeans',
            'options': ['kmeans'],
            'type': 'list',
            'value_cast': str
        },
        {
            'code': 'glue_method',
            'description': u'метод склейки микрокластеров',
            'value': 'main',
            'options': ['main', 'centroid'],
            'type': 'list',
            'value_cast': str
        },
        {
            'code': 'glue_threshold',
            'description': u'порог склейки микрокластеров',
            'value': 0.2,
            'options': [0.0, 1.0],
            'type': 'number',
            'value_cast': float
        }
    ]

    RUNS_LIST_KEY = 'runner:runs'
    RUNS_QUEUE_KEY = 'runner:queue'
    RUN_KEY = 'runner:run:{run}'
    STEP_TIME_DELTA = 30

    def __init__(self):
        super(ClusteringRunner, self).__init__(settings.REDISES_RECOMMEND)

    def _experiment_data(self, run_id, run_params, clusters, begin, end, runs=None, message='Still running'):
        if runs is None:
            runs = self.slave.lrange(self.RUNS_LIST_KEY, 0, -1)

        parameters = []
        for param in self.PARAMETERS:
            parameters.append(
                dict(param, value=run_params.get(param['code']) or param['value'])
            )

        return dict(
            run_id=run_id,
            parameters=parameters,
            clusters=clusters or [],
            runs=runs or [],
            message=not clusters and run_id and message or '',
            begin=begin,
            end=end
        )

    def get(self, run_id=None):
        runs = self.slave.lrange(self.RUNS_LIST_KEY, 0, -1) or []
        if not run_id and runs:
            run_id = runs[0]

        run = run_id and self.slave.hgetall(self.RUN_KEY.format(run=run_id)) or {}
        clusters = json.loads(run.get('clusters', '[]'))
        parameters = json.loads(run.get('parameters', '{}'))
        return self._experiment_data(
            run_id,
            parameters,
            clusters,
            run.get('begin'),
            run.get('end'),
            runs,
            run.get('message')
        )

    def make_experiment(self, request):
        now = dt.datetime.now()
        run_id=now.strftime('%d.%m.%y_%H:%M')
        parameters = {}
        for param in self.PARAMETERS:
            parameters[param['code']] = param['value_cast'](request.POST.get(param['code']))

        begin = request.POST.get('from') or (now - dt.timedelta(days=1)).strftime('%d.%m.%Y')
        end = request.POST.get('to') or now.strftime('%d.%m.%Y')

        clusters = []
        run_key = self.RUN_KEY.format(run=run_id)
        self.master.hmset(run_key,
            dict(
                parameters=json.dumps(parameters),
                clusters=json.dumps(clusters),
                begin=begin,
                end=end
            )
        )
        self.master.lpush(self.RUNS_LIST_KEY, run_id)
        self.master.rpush(self.RUNS_QUEUE_KEY, run_id)

        return self._experiment_data(run_id, parameters, clusters, begin, end)

    def run_experiment(self):
        from scraper.clusterizer.lemmatize import lemmatize_twolevel
        from scraper.clusterizer.boiler import boil_macro
        from scraper.redis import ClusterIdSeq
        from scraper.clusterizer import docspace
        from scraper.clusterizer.baker_storage import AllClustersStorage

        run_id = self.master.lpop(self.RUNS_QUEUE_KEY)
        if not run_id:
            return

        run_key = self.RUN_KEY.format(run=run_id)
        run = self.slave.hgetall(run_key)
        params = json.loads(run['parameters'])
        params = {param.upper(): value for param, value in params.iteritems()}
        config.update(**dict(params, EXEC='direct'))

        ClusterIdSeq().set_start_value(0)
        AllClustersStorage().clean_storage()
        clusters = AllClustersStorage()

        item_dict = {}
        begin = dt.datetime.strptime(run['begin'], '%d.%m.%Y')
        end = dt.datetime.strptime(run['end'], '%d.%m.%Y')
        batch_id = 0
        while begin < end:
            items = db.slave_session.query(models.Item).filter(
                (models.Item.status == models.Item.STATUS_BAKED) &
                (models.Item.ctime > begin) &
                (models.Item.ctime <= begin + dt.timedelta(minutes=self.STEP_TIME_DELTA))
            ).options(
                load_only('id', 'pubdate', 'url', 'title', 'body'),
                joinedload('resource').load_only('title')
            )

            terms = {}
            for item in items.all():
                terms[item.id] = lemmatize_twolevel(item.title, item.body)
                item_dict[item.id] = item

            added_docs, updated_docs, removed_docids, macro_updated = docspace.prepare_docs(
                clusters=clusters,
                item_ids=list(item_dict.keys()),
                clean=False,
                baker_data={}
            )

            boil_macro(clusters, added_docs + updated_docs, macro_updated)
            clusters.sync()

            elapsed_time = dt.timedelta(minutes=self.STEP_TIME_DELTA) * (batch_id + 1)
            self.master.hset(
                run_key,
                'message',
                'Still running... {:.2f} hours processed'.format(float(elapsed_time.total_seconds()) / 3600)
            )

            begin += dt.timedelta(minutes=self.STEP_TIME_DELTA)
            batch_id += 1

        macro_clusters = []
        for macro in clusters.itervalues():
            cluster = []
            for micro_id, micro in macro.iteritems():
                for doc in micro:
                    item = item_dict[doc.id]
                    cluster.append(
                        dict(
                            title=item.title,
                            link=item.url,
                            pubdate=item.pubdate.strftime('%d.%m.%y %H:%M'),
                            resource_name=item.resource.title_short,
                            micro_cluster_id=micro_id
                        )
                    )
            macro_clusters.append(dict(docs=cluster))

        self.master.hset(
            run_key, 'clusters', json.dumps(macro_clusters)
        )

