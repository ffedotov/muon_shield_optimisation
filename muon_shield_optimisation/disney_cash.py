#!/usr/bin/env python3
import time
import argparse
import json
import pickle
import csv
from disneylandClient import (Job, RequestWithId, ListJobsRequest, new_client)
from disney_oneshot import (get_result, CreateJobInput, CreateMetaData,
                            ExtractParams, STATUS_FINAL)

from config import (RUN, POINTS_IN_BATCH, RANDOM_STARTS, MIN, IMAGE_TAG,
                    COMPATIBLE_TAGS)

SLEEP_TIME = 60  # seconds


def ProcessPoint(jobs, tag):
    if json.loads(jobs[0].metadata)['user']['tag'] == tag:
        try:
            weight, length, _, muons_w = get_result(jobs)
            y = FCN(weight, muons_w, length)
            X = ExtractParams(jobs[0].metadata)

            stub.CreateJob(
                Job(input='',
                    output=str(y),
                    kind='point',
                    metadata=jobs[0].metadata))
            # TODO modify original jobs to mark them as processed,
            # job_id of point
            print(X, y)
            return X, y
        except Exception as e:
            print(e)





def WaitCompleteness(jobs):

    work_time = 0
    while True:
        time.sleep(SLEEP_TIME)

        ids = [[job.id for job in point] for point in jobs]
        uncompleted_jobs = [[
            stub.GetJob(RequestWithId(id=id)) for id in point
        ] for point in ids]
        jobs_completed = [
            job.status in STATUS_FINAL
            for point in uncompleted_jobs for job in point
        ]

        if all(jobs_completed):
            return uncompleted_jobs

        print('[{}] Waiting...'.format(time.time()))
        work_time += 60

        if work_time > 60 * 60 * 3:
            completed_jobs = []
            for point in uncompleted_jobs:
                if all([job.status in STATUS_FINAL for job in point]):
                    completed_jobs.append(point)

            return completed_jobs


def ProcessJobs(jobs, tag):
    print('[{}] Processing jobs...'.format(time.time()))
    results = [ProcessPoint(point, tag) for point in jobs]
    print(f'Got results {results}')
    results = [result for result in results if result]
    return zip(*results) if results else ([], [])


stub = new_client()

cache = {
    # id: loss
}


def SubmitDockerJobs(point, tag, sampling, seed):
    return [
        stub.CreateJob(
            Job(input=CreateJobInput(point, i, sampling=sampling, seed=seed),
                kind='docker',
                metadata=CreateMetaData(
                    point, tag, sampling=sampling, seed=seed)))
        for i in range(16)
    ]


def ProcessPoints(points):
    X = []
    y = []

    for point in points:
        try:
            X.append(ExtractParams(point.metadata))
            y.append(float(point.output))
        except Exception as e:
            print(e)
            raise

    return X, y


def FilterPoints(points, seed, sampling, image_tag=IMAGE_TAG, tag='all'):
    filtered = []
    for point in points:
        if len(ExtractParams(point.metadata)) != 56:
            continue
        metadata = json.loads(point.metadata)['user']
        if ((tag == 'all' or metadata['tag'] == tag)
                and metadata['image_tag'] == image_tag
                and (metadata['seed'] == seed or seed == 'all')
                and (metadata['sampling'] == sampling or sampling == 'all')):
            filtered.append(point)
    return filtered


def CalculatePoints(points, tag, sampling, seed):
    shield_jobs = [
        SubmitDockerJobs(point, tag, sampling=sampling, seed=seed)
        for point in points if json.dumps(point) not in cache
    ]

    X_cached, y_cached = zip(*[(point, cache[json.dumps(point)])
                               for point in points
                               if json.dumps(point) in cache])

    if shield_jobs:
        shield_jobs = WaitCompleteness(shield_jobs)
        X_new, y_new = ProcessJobs(shield_jobs, tag)

    X, y = X_cached + X_new, y_cached + y_new

    return X, y


def main():
    parser = argparse.ArgumentParser(description='Start optimizer.')
    parser.add_argument('--opt', help='Write an optimizer.', default='rf')
    parser.add_argument('--tag', help='Additional suffix for tag', default='')
    parser.add_argument(
        '--state', help='Random state of Optimizer', default=None)
    parser.add_argument('--seed', help='Random seed of simulation', default=1)
    parser.add_argument('--sampling', default=37)
    parser.add_argument('--reduced', action='store_true')
    args = parser.parse_args()
    tag = f'{RUN}_{args.opt}' + f'_{args.tag}' if args.tag else ''

    # TODO use random points for init, don't tag them with optimiser

    all_points = stub.ListJobs(ListJobsRequest(kind='point', how_many=0)).jobs
    X, y = ProcessPoints(
        FilterPoints(
            all_points, tag='all', seed=args.seed, sampling=args.sampling))
    print('Size')
    print(len(X))
    for x, loss in zip(X, y):     
         print(x, loss)

    for image in COMPATIBLE_TAGS[IMAGE_TAG]:
        for x in zip(ProcessPoints(FilterPoints(all_points,tag='all',seed=args.seed,sampling=args.sampling,image_tag=image))):
            print(x)  


if __name__ == '__main__':
    main()
