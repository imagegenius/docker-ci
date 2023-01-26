#!/usr/bin/env python3

from multiprocessing.pool import ThreadPool
from threading import current_thread
import os
import shutil
import time
import logging
import mimetypes
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError
import docker
from docker.errors import APIError
from docker.models.containers import Container
import anybadge
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from jinja2 import Environment, FileSystemLoader
from pyvirtualdisplay import Display


class SetEnvs():
    """Simple helper class that sets up the ENVs"""
    def __init__(self) -> None:
        self.logger = logging.getLogger("SetEnvs")

        os.environ['S6_VERBOSITY'] = os.environ.get("CI_S6_VERBOSITY","2")
        # Set the optional parameters
        self.dockerenv = self.convert_env(os.environ.get("DOCKER_ENV", ""))
        self.webauth = os.environ.get('WEB_AUTH', 'user:password')
        self.webpath = os.environ.get('WEB_PATH', '')
        self.screenshot = os.environ.get('WEB_SCREENSHOT', 'false')
        self.screenshot_delay = os.environ.get('WEB_SCREENSHOT_DELAY', '30')
        self.port = os.environ.get('PORT', '80')
        self.ssl = os.environ.get('SSL', 'false')
        self.region = os.environ.get('S3_REGION', 'ap-melbourne-1')
        self.bucket = os.environ.get('S3_BUCKET', 'ci-tests.imagegenius.io')
        self.test_container_delay = os.environ.get('DELAY_START', '5')
        self.check_env()


    def convert_env(self, envs:str = None) -> dict:
        """Convert env DOCKER_ENV to dictionary"""
        env_dict = {}
        if envs:
            self.logger.info("Converting envs")
            try:
                if '|' in envs:
                    for varpair in envs.split('|'):
                        var = varpair.split('=')
                        env_dict[var[0]] = var[1]
                else:
                    var = envs.split('=')
                    env_dict[var[0]] = var[1]
                env_dict["S6_VERBOSITY"] = os.environ.get('S6_VERBOSITY')
            except Exception as error:
                self.logger.exception(error)
                raise CIError(f"Failed converting DOCKER_ENV: {envs} to dictionary") from error
        return env_dict


    def check_env(self) -> None:
        """Make sure all needed ENVs are set"""
        try:
            self.image = os.environ['IMAGE']
            self.base = os.environ['BASE']
            self.s3_key = os.environ['ACCESS_KEY']
            self.s3_secret = os.environ['SECRET_KEY']
            self.meta_tag = os.environ['META_TAG']
            self.tags_env = os.environ['TAGS']
        except KeyError as error:
            self.logger.exception("Key %s is not set in ENV!", error)
            raise CIError(f'Key {error} is not set in ENV!') from error


class CI(SetEnvs):
    """CI object to use for testing image tags.

    Args:
        SetEnvs (Object): Helper class that initializes and checks that all the necessary enviroment variables exists. Object is initialized upon init of CI.
    """
    def __init__(self) -> None:
        super().__init__()  # Init the SetEnvs object.
        self.logger = logging.getLogger("IG CI")
        logging.getLogger("botocore.auth").setLevel(logging.INFO)  # Don't log the S3 authentication steps.

        self.client = docker.from_env()
        self.tags = list(self.tags_env.split('|'))
        self.tag_report_tests = {tag: {'test':{}} for tag in self.tags} # Adds all the tags as keys with an empty dict as value to the dict
        self.report_containers: dict[str,dict] = {}
        self.report_status = 'PASS'
        self.outdir = f'{os.path.dirname(os.path.realpath(__file__))}/output/{self.image}/{self.meta_tag}'
        os.makedirs(self.outdir, exist_ok=True)
        self.s3_client = boto3.Session().client(
            's3',
            region_name=self.region,
            aws_access_key_id=self.s3_key,
            aws_secret_access_key=self.s3_secret,
			endpoint_url="https://axtrmi976kgd.compat.objectstorage.ap-melbourne-1.oraclecloud.com")

    def run(self,tags: list) -> None:
        """Will iterate over all the tags running container_test() on each tag, multithreaded.

        Also does a pull of the linuxserver/tester:latest image before running container_test.

        Args:
            `tags` (list): All the tags we will test on the image.

        """
        self.logger.info("Pulling ghcr.io/linuxserver/tester:latest")
        self.client.images.pull(repository="ghcr.io/linuxserver/tester", tag="latest") # Pulls latest tester image. 
        thread_pool = ThreadPool(processes=10)
        thread_pool.map(self.container_test,tags)
        display = Display(size=(1920, 1080)) # Setup an x virtual frame buffer (Xvfb) that Selenium can use during the tests.
        display.start()
        thread_pool.close()
        thread_pool.join()
        display.stop()


    def container_test(self, tag: str) -> None:
        """Main container test logic.

        Args:
            `tag` (str): The container tag

        1. Spins up the container tag
            Checks the container logs for either `[services.d] done.` or `[ig-init] done.`
        2. Export the build version from the Container object.
        3. Export the package info from the Container object.
        4. Take a screenshot for the report.
        5. Add report information to report.json.
        """
        def _endtest(self: CI, container:Container, tag:str, build_version:str, packages:str, test_success: bool):
            """End the test with as much info as we have and append to the report.

            Args:
                `container` (Container): Container object
                `tag` (str): The container tag
                `build_version` (str): The Container build version
                `packages` (str): Package dump from the container
                `test_success` (bool): If the testing of the container failed or not
            """
            logblob = container.logs().decode('utf-8')
            container.remove(force='true')
            warning_texts = {
                "uwsgi": "This image uses uWSGI and might not start on ARM/QEMU"
            }
            # Add the info to the report
            self.report_containers[tag] = {
                'logs': logblob,
                'sysinfo': packages,
                'warnings': {
                    'uwsgi': warning_texts["uwsgi"] if "uwsgi" in packages and "arm" in tag else ""
                },
                'build_version': build_version,
                'test_results': self.tag_report_tests[tag]['test'],
                'test_success': test_success,
                }
            self.report_containers[tag]["has_warnings"] = any(warning[1] for warning in self.report_containers[tag]["warnings"].items())
        # Name the thread for easier debugging.
        if "amd" in tag or "arm" in tag:
            current_thread().name = f"{tag[:5].upper()}Thread"

        # Start the container
        self.logger.info('Starting test of: %s', tag)
        container: Container = self.client.containers.run(f'{self.image}:{tag}',
                                               detach=True,
                                               environment=self.dockerenv)
        container_config = container.attrs["Config"]["Env"]
        self.logger.info("Container config of tag %s: %s",tag,container_config)
        # Watch the logs for no more than 5 minutes
        logsfound = False
        t_end = time.time() + 60 * 5
        self.logger.info("Checking logs for the 'done' message on tag: %s",tag)
        while time.time() < t_end:
            try:
                logblob = container.logs().decode('utf-8')
                if '[services.d] done.' in logblob or '[ig-init] done.' in logblob:
                    logsfound = True
                    break
                time.sleep(1)
            except APIError as error:
                self.logger.exception('Container startup failed for %s', tag)
                self.tag_report_tests[tag]['test']['Container startup'] = (dict(sorted({
                    'status':'FAIL',
                    'message': f'INIT NOT FINISHED: {error}'
                    }.items())))                               
                self.report_status = 'FAIL'
                _endtest(self, container, tag, 'ERROR', 'ERROR', False)
                return
        # Grab build version
        try:
            self.logger.info("Fetching build version on tag: %s",tag)
            build_version = container.attrs['Config']['Labels']['build_version']
            self.tag_report_tests[tag]['test']['Get build version'] = (dict(sorted({
                'status':'PASS',
                'message':'-'}.items())))
            self.logger.info('Get build version %s: PASS', tag)
        except APIError as error:
            build_version = 'ERROR'
            self.tag_report_tests[tag]['test']['Get build version'] = (dict(sorted({
                'status':'FAIL',
                'message':error}.items())))
            self.logger.exception('Get build version %s: FAIL', tag)
            self.report_status = 'FAIL'
            _endtest(self, container, tag, build_version, 'ERROR', False)
            return
        # Check if the startup marker was found in the logs during the 2 minute spinup
        if logsfound:
            self.logger.info('Container startup completed for %s', tag)
            self.tag_report_tests[tag]['test']['Container startup'] = (dict(sorted({
                'status':'PASS',
                'message':'-'}.items())))
            self.logger.info('Container startup %s: PASS', tag)
        else:
            self.logger.error('Container startup failed for %s', tag)
            self.tag_report_tests[tag]['test']['Container startup'] = (dict(sorted({
                'status':'FAIL',
                'message':'INIT NOT FINISHED'}.items())))
            self.logger.error('Container startup %s: FAIL - INIT NOT FINISHED', tag)
            self.report_status = 'FAIL'
            _endtest(self, container, tag, build_version, 'ERROR', False)
            return
        # Dump package information
        dump_commands = {
            'alpine': 'apk info -v',
            'debian': 'apt list -qq --installed',
            'ubuntu': 'apt list -qq --installed',
            'fedora': 'rpm -qa',
            'arch': 'pacman -Q'
            }
        try:
            self.logger.info('Dumping package info for %s',tag)
            info = container.exec_run(dump_commands[self.base])
            packages = info[1].decode('utf-8')
            self.tag_report_tests[tag]['test']['Dump package info'] = (dict(sorted({
                'status':'PASS',
                'message':'-'}.items())))
            self.logger.info('Dump package info %s: PASS', tag)
        except (APIError, IndexError) as error:
            packages = 'ERROR'
            self.logger.exception(str(error))
            self.tag_report_tests[tag]['test']['Dump package info'] = (dict(sorted({
                'Dump package info':'FAIL',
                'message':error}.items())))
            self.logger.error('Dump package info %s: FAIL', tag)
            self.report_status = 'FAIL' 
            _endtest(self, container, tag, build_version, packages, False)
            return
        # Screenshot web interface and check connectivity
        if self.screenshot == 'true':
            self.take_screenshot(container, tag)
        # If all info is present end test
        _endtest(self, container, tag, build_version, packages, True)
        return


    def report_render(self) -> None:
        """Render the index file for upload"""
        self.logger.info('Rendering Report')
        env = Environment( loader = FileSystemLoader(os.path.dirname(os.path.realpath(__file__))) )
        template = env.get_template('template.html')
        self.report_containers = json.loads(json.dumps(self.report_containers,sort_keys=True))
        with open(f'{os.path.dirname(os.path.realpath(__file__))}/index.html', mode="w", encoding='utf-8') as file_:
            file_.write(template.render(
            report_containers=self.report_containers,
            report_status=self.report_status,
            meta_tag=self.meta_tag,
            image=self.image,
            bucket=self.bucket,
            region=self.region,
            screenshot=self.screenshot
            ))


    def badge_render(self) -> None:
        """Render the badge file for upload"""
        self.logger.info("Creating badge")
        try:
            badge = anybadge.Badge('CI', self.report_status, thresholds={
                                   'PASS': 'green', 'FAIL': 'red'})
            badge.write_badge(f'{self.outdir}/badge.svg')
            with open(f'{self.outdir}/ci-status.yml', 'w', encoding='utf-8') as file:
                file.write(f'CI: "{self.report_status}"')
        except (ValueError,RuntimeError,FileNotFoundError,OSError) as error:
            self.logger.exception(error)

    def json_render(self) -> None:
        """Create a JSON file of the report data."""
        self.logger.info("Creating report.json file")
        try:
            with open(f'{os.path.dirname(os.path.realpath(__file__))}/report.json', mode="w", encoding='utf-8') as file:
                json.dump(self.report_containers, file, indent=2, sort_keys=True)
            shutil.copyfile(f'{os.path.dirname(os.path.realpath(__file__))}/report.json', f'{self.outdir}/report.json')
        except (OSError,FileNotFoundError,TypeError,Exception) as error:
            self.logger.exception(error)

    def report_upload(self) -> None:
        """Upload report files to S3

        Raises:
            Exception: S3UploadFailedError
            Exception: ValueError
            Exception: ClientError
        """
        self.logger.info('Uploading report files')
        # Index file upload
        index_file = f'{os.path.dirname(os.path.realpath(__file__))}/index.html'
        shutil.copyfile(f'{os.path.dirname(os.path.realpath(__file__))}/404.jpg', f'{self.outdir}/404.jpg')
        ctype = {'ContentType': 'text/html', 'ACL': 'public-read', 'CacheControl': 'no-cache'}  # Set content type
        try:
            self.upload_file(index_file, "index.html", ctype)
        except (S3UploadFailedError, ValueError, ClientError) as error:
            self.logger.exception('Upload Error: %s',error)
            self.log_upload()
            raise CIError(f'Upload Error: {error}') from error

        # Loop through files in outdir and upload
        for filename in os.listdir(self.outdir):
            time.sleep(0.5)
            ctype = mimetypes.guess_type(filename.lower(), strict=False)
            ctype = {'ContentType': ctype[0] if ctype[0] else 'text/plain', 'ACL': 'public-read', 'CacheControl': 'no-cache'}  # Set content types for files
            try:
                self.upload_file(f'{self.outdir}/{filename}', filename, ctype)
            except (S3UploadFailedError, ValueError, ClientError) as error:
                self.logger.exception('Upload Error: %s',error)
                self.log_upload()
                raise CIError(f'Upload Error: {error}') from error
        self.logger.info('Report available on https://ci-tests.imagegenius.io/%s/index.html', f'{self.image}/{self.meta_tag}')


    def upload_file(self, file_path:str, object_name:str, content_type:dict) -> None:
        """Upload a file to an S3 bucket

        Args:
            `file_path` (str): File to upload
            `bucket` (str): Bucket to upload to
            `object_name` (str): S3 object name.
        """
        self.logger.info('Uploading %s to %s bucket',file_path, self.bucket)
        destination_dir = f'{self.image}/{self.meta_tag}'
        latest_dir = f'{self.image}/latest'
        self.s3_client.upload_file(file_path, self.bucket, f'{destination_dir}/{object_name}', ExtraArgs=content_type)
        self.s3_client.upload_file(file_path, self.bucket, f'{latest_dir}/{object_name}', ExtraArgs=content_type)

    def log_upload(self) -> None:
        """Upload the ci.log to S3

        Raises:
            Exception: S3UploadFailedError
            Exception: ClientError
        """
        self.logger.info('Uploading logs')
        try:
            self.upload_file("/ci.log", 'ci.log', {'ContentType': 'text/plain', 'ACL': 'public-read'}) 
        except (S3UploadFailedError, ClientError) as error:
            self.logger.exception('Upload Error: %s',error)


    def take_screenshot(self, container: Container, tag:str) -> None:
        """Take a screenshot and save it to self.outdir

        Spins up an ghcr.io/linuxserver/tester container and takes a screenshot using Selenium.

        Args:
            `container` (Container): Container object
            `tag` (str): The container tag we are testing.
        """
        proto = 'https' if self.ssl.upper() == 'TRUE' else 'http'
        # Sleep for the user specified amount of time
        self.logger.info('Sleeping for %s seconds before reloading container: %s and refreshing container attrs', self.test_container_delay, container.image)
        time.sleep(int(self.test_container_delay))
        container.reload()
        ip_adr = container.attrs['NetworkSettings']['Networks']['bridge']['IPAddress']
        endpoint = f'{proto}://{self.webauth}@{ip_adr}:{self.port}{self.webpath}'
        testercontainer, test_endpoint = self.start_tester(proto,endpoint,tag)
        try:
            driver = self.setup_driver()
            driver.get(test_endpoint)
            self.logger.info('Sleeping for %s seconds before creating a screenshot on %s', self.screenshot_delay, tag)
            time.sleep(int(self.screenshot_delay))
            self.logger.info('Taking screenshot of %s at %s', tag, endpoint)
            driver.get_screenshot_as_file(f'{self.outdir}/{tag}.png')
            self.tag_report_tests[tag]['test']['Get screenshot'] = (dict(sorted({
                'status':'PASS',
                'message':'-'}.items())))
            self.logger.info('Screenshot %s: PASS', tag)
        except (requests.Timeout, requests.ConnectionError, KeyError) as error:
            self.tag_report_tests[tag]['test']['Get screenshot'] = (dict(sorted({
                'status':'FAIL',
                'message': f'CONNECTION ERROR: {error}'}.items())))
            self.logger.exception('Screenshot %s FAIL CONNECTION ERROR', tag)
        except TimeoutException as error:
            self.tag_report_tests[tag]['test']['Get screenshot'] = (dict(sorted({
                'status':'FAIL',
                'message':f'TIMEOUT: {error}'}.items())))
            self.logger.exception('Screenshot %s FAIL TIMEOUT', tag)
        except (WebDriverException, Exception) as error:
            self.tag_report_tests[tag]['test']['Get screenshot'] = (dict(sorted({
                'status':'FAIL',
                'message':f'UNKNOWN: {error}'}.items())))
            self.logger.exception('Screenshot %s FAIL UNKNOWN: %s', tag, error)
        finally:
            testercontainer.remove(force='true')


    def start_tester(self, proto:str, endpoint:str, tag:str) -> tuple[Container,str]:
        """Spin up an RDP test container to load the container web ui.

        Args:
            `proto` (str): The protocol to use for the endpoint.
            `endpoint` (str): The container endpoint to use with the tester container.
            `tag` (str): The container tag

        Returns:
            Container/str: Returns the tester Container object and the tester endpoint
        """
        self.logger.info("Starting tester container for tag: %s", tag)
        testercontainer: Container = self.client.containers.run('ghcr.io/linuxserver/tester:latest',
                                                     shm_size='1G',
                                                     security_opt=["seccomp=unconfined"],
                                                     detach=True,
                                                     environment={'URL': endpoint})
        #Sleep for the user specified amount of time
        self.logger.info('Sleeping for %s seconds before reloading %s and refreshing container attrs on %s run', self.test_container_delay, testercontainer.image, tag)
        time.sleep(int(self.test_container_delay))
        testercontainer.reload()
        testerip = testercontainer.attrs['NetworkSettings']['Networks']['bridge']['IPAddress']
        testerendpoint = f'http://{testerip}:3000'
        session = requests.Session()
        retries = Retry(total=10, backoff_factor=2,status_forcelist=[502, 503, 504])
        session.mount(proto, HTTPAdapter(max_retries=retries))
        session.get(testerendpoint)
        return testercontainer, testerendpoint


    def setup_driver(self) -> webdriver.Chrome:
        """Return a single ChromiumDriver object the class can use

        Returns:
            Webdriver: Returns a Chromedriver object
        """
        self.logger.info("Init Chromedriver")
        # Selenium webdriver options
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920x1080')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--disable-dev-shm-usage')  # https://developers.google.com/web/tools/puppeteer/troubleshooting#tips
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(60)
        return driver

class CIError(Exception):
    pass
