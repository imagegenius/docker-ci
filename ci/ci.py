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
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError
from PIL import Image
import docker
from docker.errors import APIError, ContainerError, ImageNotFound
from docker.models.containers import Container
import anybadge
from ansi2html import Ansi2HTMLConverter
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from jinja2 import Environment, FileSystemLoader, select_autoescape


class SetEnvs():
    """Simple helper class that sets up the ENVs"""

    def __init__(self) -> None:
        self.logger = logging.getLogger("SetEnvs")

        os.environ['S6_VERBOSITY'] = os.environ.get("CI_S6_VERBOSITY", "2")
        # Set the optional parameters
        self.dockerenv = self.convert_env(os.environ.get("DOCKER_ENV", ""))
        self.webauth = os.environ.get('WEB_AUTH', 'user:password')
        self.webpath = os.environ.get('WEB_PATH', '')
        self.branch = os.environ.get('BRANCH', '')
        self.screenshot = os.environ.get('WEB_SCREENSHOT', 'false')
        self.logs_delay = os.environ.get('DOCKER_LOGS_DELAY', '300')
        self.startup_delay = os.environ.get('STARTUP_DELAY', '10')
        self.port = os.environ.get('PORT', '80')
        self.ssl = os.environ.get('SSL', 'false')
        self.region = os.environ.get('S3_REGION', 'ap-melbourne-1')
        self.bucket = os.environ.get('S3_BUCKET', 'ci-tests.imagegenius.io')
        self.check_env()

    def convert_env(self, envs: str = None) -> dict:
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
                self.logger.exception(
                    "Failed to convert DOCKER_ENV: %s to dictionary!", envs)
                raise CIError(
                    f"Failed converting DOCKER_ENV: {envs} to dictionary") from error
        return env_dict

    def check_env(self) -> None:
        """Make sure all needed ENVs are set"""
        try:
            self.image = os.environ['IMAGE']
            self.container = os.environ['CONTAINER']
            self.base = os.environ['BASE']
            self.s3_key = os.environ['ACCESS_KEY']
            self.s3_secret = os.environ['SECRET_KEY']
            self.meta_tag = os.environ['META_TAG']
            self.tags_env = os.environ['TAGS']
        except KeyError as error:
            self.logger.exception("Key is not set in ENV!")
            raise CIError(f'Key {error} is not set in ENV!') from error


class CI(SetEnvs):
    """CI object to use for testing image tags.

    Args:
        SetEnvs (Object): Helper class that initializes and checks that all the necessary enviroment variables exists. Object is initialized upon init of CI.
    """

    def __init__(self) -> None:
        super().__init__()  # Init the SetEnvs object.
        self.logger = logging.getLogger("IG CI")
        # Don't log the S3 authentication steps.
        logging.getLogger("botocore.auth").setLevel(logging.INFO)

        self.client = docker.from_env()
        self.tags = list(self.tags_env.split('|'))
        # Adds all the tags as keys with an empty dict as value to the dict
        self.tag_report_tests = {tag: {'test': {}} for tag in self.tags}
        self.report_containers: dict[str, dict] = {}
        self.report_status = 'PASS'
        self.outdir = f'{os.path.dirname(os.path.realpath(__file__))}/output/{self.container}/{self.meta_tag}'
        os.makedirs(self.outdir, exist_ok=True)
        self.s3_client = boto3.Session().client(
            's3',
            region_name=self.region,
            aws_access_key_id=self.s3_key,
            aws_secret_access_key=self.s3_secret,
            endpoint_url="https://s3.imagegenius.io")

    def run(self, tags: list) -> None:
        """Will iterate over all the tags running container_test() on each tag, multithreaded.


        Args:
            `tags` (list): All the tags we will test on the image.

        """
        thread_pool = ThreadPool(processes=10)
        thread_pool.map(self.container_test, tags)
        thread_pool.close()
        thread_pool.join()

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
        # Name the thread for easier debugging.
        current_thread().name = f"{self.get_platform(tag).upper()}Thread"

        # Start the container
        self.logger.info('Starting test of: %s', tag)
        container: Container = self.client.containers.run(f'{self.image}:{tag}',
                                                          detach=True,
                                                          environment=self.dockerenv)
        container_config = container.attrs["Config"]["Env"]
        self.logger.info("Container config of tag %s: %s",
                         tag, container_config)

        # Watch the logs for no more than 5 minutes
        logsfound = self.watch_container_logs(container, tag)
        if not logsfound:
            self._endtest(container, tag, "ERROR", "ERROR", False)
            return

        build_version = self.get_build_version(
            container, tag)  # Get the image build version
        if build_version == "ERROR":
            self._endtest(container, tag, build_version, "ERROR", False)
            return

        sbom = self.generate_sbom(tag)
        if sbom == "ERROR":
            self._endtest(container, tag, build_version, sbom, False)
            return

        # Screenshot web interface and check connectivity
        if self.screenshot == 'true':
            self.take_screenshot(container, tag)

        self._endtest(container, tag, build_version, sbom, True)
        self.logger.info("Testing of %s PASSED", tag)
        return

    def _endtest(self: 'CI', container: Container, tag: str, build_version: str, packages: str, test_success: bool) -> None:
        """End the test with as much info as we have and append to the report.

        Args:
            `container` (Container): Container object
            `tag` (str): The container tag
            `build_version` (str): The Container build version
            `packages` (str): SBOM dump from the container
            `test_success` (bool): If the testing of the container failed or not
        """
        logblob = container.logs().decode('utf-8')
        # Generate html container log file based on the latest logs
        self.create_html_ansi_file(logblob, tag, "log")
        try:
            container.remove(force='true')
        except APIError:
            self.logger.exception("Failed to remove container %s", tag)
        warning_texts = {
            "dotnet": "May be a .NET app. Service might not start on ARM32 with QEMU",
            "uwsgi": "This image uses uWSGI and might not start on ARM/QEMU"
        }
        # Add the info to the report
        self.report_containers[tag] = {
            'logs': logblob,
            'sysinfo': packages,
            'warnings': {
                'dotnet': warning_texts["dotnet"] if "icu-libs" in packages and "arm32" in tag else "",
                'uwsgi': warning_texts["uwsgi"] if "uwsgi" in packages and "arm" in tag else ""
            },
            'build_version': build_version,
            'test_results': self.tag_report_tests[tag]['test'],
            'test_success': test_success,
        }
        self.report_containers[tag]["has_warnings"] = any(
            warning[1] for warning in self.report_containers[tag]["warnings"].items())

    def get_platform(self, tag: str) -> str:
        """Check the 5 first characters of the tag and return the platform.

        If no match is found return amd64.

        Returns:
            str: The platform
        """
        platform = tag[:5]
        match platform:
            case "amd64":
                return "amd64"
            case "arm64":
                return "arm64"
            case "arm32":
                return "arm"
            case _:
                return "amd64"

    def generate_sbom(self, tag: str) -> str:
        """Generate the SBOM for the image tag.

        Creates the output file in `{self.outdir}/{tag}.sbom.html`

        Args:
            tag (str): The tag we are testing

        Returns:
            bool: Return the output if successful otherwise "ERROR".
        """
        platform = self.get_platform(tag)
        syft: Container = self.client.containers.run(
            image="ghcr.io/anchore/syft:v0.76.1",
            command=f"{self.image}:{tag} --platform=linux/{platform}",
            detach=True,
            volumes={
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "ro"}
            }
        )
        self.logger.info('Creating SBOM package list on %s', tag)

        t_end = time.time() + int(self.logs_delay)
        self.logger.info(
            "Tailing the syft container logs for %s seconds looking the 'VERSION' message on tag: %s", self.logs_delay, tag)
        error_message = "Did not find the 'VERSION' keyword in the syft container logs"
        while time.time() < t_end:
            time.sleep(5)
            try:
                logblob = syft.logs().decode('utf-8')
                if 'VERSION' in logblob:
                    self.logger.info(
                        'Get package versions for %s completed', tag)
                    self.tag_report_tests[tag]['test']['Create SBOM'] = (dict(sorted({
                        'status': 'PASS',
                        'message': '-'}.items())))
                    self.logger.info('Create SBOM package list %s: PASS', tag)
                    self.create_html_ansi_file(str(logblob), tag, "sbom")
                    try:
                        syft.remove(force=True)
                    except Exception:
                        self.logger.exception(
                            "Failed to remove the syft container, %s", tag)
                    return logblob
            except (APIError, ContainerError, ImageNotFound) as error:
                error_message = error
                self.logger.exception(
                    'Creating SBOM package list on %s: FAIL', tag)
        self.logger.error(
            "Failed to generate SBOM output on tag %s. SBOM output:\n%s", tag, logblob)
        self.report_status = "FAIL"
        self.tag_report_tests[tag]['test']['Create SBOM'] = (dict(sorted({
            "Create SBOM": "FAIL",
            "message": str(error_message)}.items())))
        try:
            syft.remove(force=True)
        except Exception:
            self.logger.exception(
                "Failed to remove the syft container, %s", tag)
        return "ERROR"

    def get_build_version(self, container: Container, tag: str) -> str:
        """Fetch the build version from the container object attributes.

        Args:
            container (Container): The container we are testing
            tag (str): The current tag we are testing

        Returns:
            str: Returns the build version or 'ERROR'
        """
        try:
            self.logger.info("Fetching build version on tag: %s", tag)
            build_version = container.attrs['Config']['Labels']['build_version']
            self.tag_report_tests[tag]['test']['Get build version'] = (dict(sorted({
                'status': 'PASS',
                'message': '-'}.items())))
            self.logger.info('Get build version on tag "%s": PASS', tag)
        except (APIError, KeyError) as error:
            self.logger.exception('Get build version on tag "%s": FAIL', tag)
            build_version = 'ERROR'
            if isinstance(error, KeyError):
                error = f"KeyError: {error}"
            self.tag_report_tests[tag]['test']['Get build version'] = (dict(sorted({
                'status': 'FAIL',
                'message': str(error)}.items())))
            self.report_status = 'FAIL'
        return build_version

    def watch_container_logs(self, container: Container, tag: str) -> bool:
        """Tail the container logs for 5 minutes and look for the init done message that tells us the container started up
        successfully.

        Args:
            container (Container): The container we are testing
            tag (str): The tag we are testing

        Returns:
            bool: Return True if the 'done' message is found, otherwise False.
        """
        t_end = time.time() + int(self.logs_delay)
        self.logger.info(
            "Tailing the %s logs for %s seconds looking for the 'done' message", tag, self.logs_delay)
        while time.time() < t_end:
            try:
                logblob = container.logs().decode('utf-8')
                if '[services.d] done.' in logblob or '[ig-init] done.' in logblob:
                    self.logger.info(
                        "Sleeping for %s seconds while %s finishes starting", self.startup_delay, tag)
                    time.sleep(int(self.startup_delay))
                    self.logger.info('Container startup completed for %s', tag)
                    self.tag_report_tests[tag]['test']['Container startup'] = (dict(sorted({
                        'status': 'PASS',
                        'message': '-'}.items())))
                    self.logger.info('Container startup %s: PASS', tag)
                    return True
                time.sleep(1)
            except APIError as error:
                self.logger.exception(
                    'Container startup %s: FAIL - INIT NOT FINISHED', tag)
                self.tag_report_tests[tag]['test']['Container startup'] = (dict(sorted({
                    'status': 'FAIL',
                    'message': f'INIT NOT FINISHED: {str(error)}'
                }.items())))
                self.report_status = 'FAIL'
                return False
        self.logger.error('Container startup failed for %s', tag)
        self.tag_report_tests[tag]['test']['Container startup'] = (dict(sorted({
            'status': 'FAIL',
            'message': 'INIT NOT FINISHED'}.items())))
        self.logger.error(
            'Container startup %s: FAIL - INIT NOT FINISHED', tag)
        self.report_status = 'FAIL'
        return False

    def report_render(self) -> None:
        """Render the index file for upload"""
        self.logger.info('Rendering Report')
        env = Environment(autoescape=select_autoescape(enabled_extensions=('html', 'xml'), default_for_string=True),
                          loader=FileSystemLoader(os.path.dirname(os.path.realpath(__file__))))
        template = env.get_template('template.html')
        self.report_containers = json.loads(
            json.dumps(self.report_containers, sort_keys=True))
        with open(f'{self.outdir}/index.html', mode="w", encoding='utf-8') as file_:
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
        except (ValueError, RuntimeError, FileNotFoundError, OSError):
            self.logger.exception("Failed to render badge file!")

    def json_render(self) -> None:
        """Create a JSON file of the report data."""
        self.logger.info("Creating report.json file")
        try:
            with open(f'{self.outdir}/report.json', mode="w", encoding='utf-8') as file:
                json.dump(self.report_containers, file,
                          indent=2, sort_keys=True)
        except (OSError, FileNotFoundError, TypeError, Exception):
            self.logger.exception("Failed to render JSON file!")

    def report_upload(self) -> None:
        """Upload report files to S3

        Raises:
            Exception: S3UploadFailedError
            Exception: ValueError
            Exception: ClientError
        """
        self.logger.info('Uploading report files')

        # Loop through files in outdir and upload
        for filename in os.listdir(self.outdir):
            time.sleep(0.5)
            ctype = mimetypes.guess_type(filename.lower(), strict=False)
            # Set content types for files
            ctype = {'ContentType': ctype[0] if ctype[0] else 'text/plain'}
            try:
                self.upload_file(f'{self.outdir}/{filename}', filename, ctype)
            except (S3UploadFailedError, ValueError, ClientError) as error:
                self.logger.exception('Upload Error!')
                self.log_upload()
                raise CIError(f'Upload Error: {error}') from error
        self.logger.info(
            'Report available on https://ci-tests.imagegenius.io/%s/index.html', f'{self.container}/{self.meta_tag}')

    def create_html_ansi_file(self, blob: str, tag: str, name: str) -> None:
        """Creates an HTML file in the 'self.outdir' directory that we upload to S3

        Args:
            blob (str): The blob you want to convert
            tag (str): The tag we are testing
            name (str): The name of the file. File name will be `{tag}.{name}.html`
        """
        try:
            self.logger.info(f"Creating {tag}.{name}.html")
            converter = Ansi2HTMLConverter()
            html_logs = converter.convert(blob)
            with open(f'{self.outdir}/{tag}.{name}.html', 'w', encoding='utf-8') as file:
                file.write(html_logs)
        except Exception:
            self.logger.exception("Failed to create %s.%s.html", tag, name)

    def upload_file(self, file_path: str, object_name: str, content_type: dict) -> None:
        """Upload a file to an S3 bucket

        Args:
            `file_path` (str): File to upload
            `bucket` (str): Bucket to upload to
            `object_name` (str): S3 object name.
        """
        self.logger.info('Uploading %s', file_path)
        destination_dir = f'{self.container}/{self.meta_tag}'
        latest_dir = f'{self.container}/latest-{self.branch}'
        self.s3_client.upload_file(
            file_path, self.bucket, f'{destination_dir}/{object_name}', ExtraArgs=content_type)
        if object_name == 'index.html' or object_name == 'ci-status.yml':
            self.s3_client.upload_file(
                file_path, self.bucket, f'{latest_dir}/{object_name}', ExtraArgs=content_type)

    def log_upload(self) -> None:
        """Upload the ci.log to S3

        Raises:
            Exception: S3UploadFailedError
            Exception: ClientError
        """
        self.logger.info('Uploading logs')
        try:
            with open(f"{self.outdir}/ci.log", "r", encoding='utf-8') as logs:
                blob = logs.read()
                self.create_html_ansi_file(blob, "python", "log")
                with open(f"{self.outdir}/ci.log", "w", encoding='utf-8') as logs:
                    # remove ANSI color codes
                    logs.write(
                        re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', blob))
                self.upload_file(f"{self.outdir}/python.log.html", 'python.log.html', {
                                 'ContentType': 'text/html'})
                self.upload_file(f"{self.outdir}/ci.log", 'ci.log',
                                 {'ContentType': 'text/plain'})
        except (S3UploadFailedError, ClientError):
            self.logger.exception('Failed to upload the CI logs!')

    def take_screenshot(self, container: Container, tag: str) -> None:
        """Take a screenshot and save it to self.outdir

        Takes a screenshot using Selenium.

        Args:
            `container` (Container): Container object
            `endpoint` (str): The endpoint to take a screenshot of.
            `tag` (str): The container tag we are testing.
        """
        proto = 'https' if self.ssl.upper() == 'TRUE' else 'http'
        try:
            container_info = self.client.api.inspect_container(container.id)
            ip_adr = container_info['NetworkSettings']['IPAddress']
            endpoint = f'{proto}://{self.webauth}@{ip_adr}:{self.port}{self.webpath}'
            driver = self.setup_driver()
            driver.get(endpoint)
            self.logger.info('Taking screenshot of %s at %s', tag, endpoint)
            driver.get_screenshot_as_file(f'{tag}.png')
            # Compress and convert the screenshot to JPEG
            im = Image.open(f'{tag}.png').convert("RGB")
            im.save(f'{self.outdir}/{tag}.jpg', 'JPEG', quality=60)
            self.tag_report_tests[tag]['test']['Get screenshot'] = (dict(sorted({
                'status': 'PASS',
                'message': '-'}.items())))
            self.logger.info('Screenshot %s: PASS', tag)
        except (requests.Timeout, requests.ConnectionError, KeyError) as error:
            self.tag_report_tests[tag]['test']['Get screenshot'] = (dict(sorted({
                'status': 'FAIL',
                'message': f'CONNECTION ERROR: {str(error)}'}.items())))
            self.logger.exception('Screenshot %s FAIL CONNECTION ERROR', tag)
        except TimeoutException as error:
            self.tag_report_tests[tag]['test']['Get screenshot'] = (dict(sorted({
                'status': 'FAIL',
                'message': f'TIMEOUT: {str(error)}'}.items())))
            self.logger.exception('Screenshot %s FAIL TIMEOUT', tag)
        except (WebDriverException, Exception) as error:
            self.tag_report_tests[tag]['test']['Get screenshot'] = (dict(sorted({
                'status': 'FAIL',
                'message': f'UNKNOWN: {str(error)}'}.items())))
            self.logger.exception('Screenshot %s FAIL UNKNOWN', tag)
        finally:
            driver.quit()

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
        # https://developers.google.com/web/tools/puppeteer/troubleshooting#tips
        chrome_options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(60)
        return driver


class CIError(Exception):
    pass
