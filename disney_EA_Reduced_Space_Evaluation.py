#!/usr/bin/env python3
import time
import argparse
import json
import pickle
import csv
from disneylandClient import (Job, RequestWithId, ListJobsRequest, new_client)
from sklearn.ensemble import GradientBoostingRegressor
from disney_common import (FCN)
from disney_oneshot import (get_result, CreateJobInput, CreateMetaData,ExtractParams, STATUS_FINAL)
from config import (RUN, POINTS_IN_BATCH, RANDOM_STARTS, MIN, IMAGE_TAG,COMPATIBLE_TAGS)

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

        if work_time > 60 * 60 * 5:
            completed_jobs = []
            for point in uncompleted_jobs:
                if all([job.status in STATUS_FINAL for job in point]):
                    completed_jobs.append(point)

            return completed_jobs


def ProcessJobs(jobs, tag):
    print('[{}] Processing jobs...'.format(time.time()))
    results = [ProcessPoint(jobs[point], tag) for point in range(len(jobs))]
    print('Got results {results}')
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


def ReadPoints(n):
 with open('/eos/experiment/ship/user/ffedship/EA/EA_ReducedSpace_cash.csv') as csv_cash_file:
  with open('/eos/experiment/ship/user/ffedship/EA/EA_ReducedSpace.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    csv_reader_cash = csv.reader(csv_cash_file, delimiter=',')
    line=0
    candidates=[]
    candidate=[]
    veteran=[]
    veterans=[]
    for row in csv_reader:
      for i in range (0,56):
       candidate.append(int(row[i]))
      candidates.append(candidate)
      candidate=[]
    print('Candidates are successfully loaded')
    csv_file.close()

    for row in csv_reader_cash:
      for j in range (0,56):
       veteran.append(int(round(float(row[j]))))
      veterans.append(veteran)
      veteran=[]    
    print('Veterans are successfully loaded')
    csv_cash_file.close()
    population=[]
    line=0
    for a in range(len(candidates)):
        cash_hit=False
        for b in range(len(veterans)):
            if candidates[a]==veterans[b]:
               cash_hit=True
        if cash_hit==False:
            population.append(candidates[a])
            line+=1
            cash_hit=False
        if line>n:
            break
        else:
            continue
 veterans=[]
 candidates=[]
 print ('Individuals selected')
 return population 

def ReadCashPoints():
 with open('/eos/experiment/ship/user/ffedship/EA/EA_ReducedSpace_cash.csv') as csv_cash_file:
    csv_reader_cash = csv.reader(csv_cash_file, delimiter=',')
    veterans=[]
    veteran=[]
    for row in csv_reader_cash:
      for i in range (0,57):
       veteran.append(float(row[i]))
      veterans.append(veteran)
      veteran=[]
 csv_cash_file.close()
 return veterans 

def PreReadCashPoints():
 with open('/eos/experiment/ship/user/ffedship/EA/EA_ReducedSpace_cash.csv') as csv_cash_file:
    csv_reader_cash = csv.reader(csv_cash_file, delimiter=',')
    veterans=[]
    veteran=[]
    for row in csv_reader_cash:
      for i in range (0,56):
       veteran.append(float(row[i]))
      veterans.append(veteran)
      veteran=[]
 csv_cash_file.close()
 return veterans
 
def CalculatePoints(points, tag, sampling, seed):
    shield_jobs = [
        SubmitDockerJobs(points[point], tag, sampling=sampling, seed=seed)
        for point in range(len(points)) 
    ]
    shield_jobs = WaitCompleteness(shield_jobs)
    X_new, y_new = ProcessJobs(shield_jobs, tag)
    X, y = X_new, y_new
    return X, y


def main():
  parser = argparse.ArgumentParser(description='Start optimizer.')
  parser.add_argument('--opt', help='Write an optimizer.', default='rf')
  parser.add_argument('--tag', help='Additional suffix for tag', default='')
  parser.add_argument('--state', help='Random state of Optimizer', default=None)
  parser.add_argument('--seed', help='Random seed of simulation', default=1)
  parser.add_argument('--sampling', default=37)
  parser.add_argument('--reduced', action='store_true')
  args = parser.parse_args()
  tag = f'{RUN}_{args.opt}' + f'_{args.tag}' if args.tag else ''
 
#Running 1000 iterations evaluating 10 point each time
  for iterator in range (0,1000):
     print ('Starting iteration')
     print (iterator)
     X_old=[]
     X_old=ReadCashPoints()
#Reading points that have been evaluated earlier         
     print('Reading cash:')
#Sending points for evaluation
     X_new, y_new = CalculatePoints(ReadPoints(10), tag, sampling=37, seed=1)
     Cash_output=open('/eos/experiment/ship/user/ffedship/EA/EA_ReducedSpace_cash.csv',"w")
     Cash_writer = csv.writer(Cash_output)
     for i in range(len(X_new)):
       X_new[i].append(y_new[i])
       Cash_writer.writerow(X_new[i])
     print ('Wrote new stuff')
#Writing New points into cache csv file
     for j in range(len(X_old)):
       Cash_writer.writerow(X_old[j])
     print ('Wrote old stuff')
     Cash_output.close()
#Saving cache

if __name__ == '__main__':
    main()
