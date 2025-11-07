"""Unit tests for data downloader module."""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from src.training_api.data.downloader import DataDownloader


class TestDataDownloader:
    """Test cases for DataDownloader class."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = DataDownloader(dest_folder=tmpdir)
            
            assert downloader.base_url == "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata"
            assert downloader.dest_folder == tmpdir
            assert downloader.years_to_download == ["2010", "2011"]
            assert os.path.exists(tmpdir)
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_url = "https://example.com/data"
            custom_years = ["2015", "2016"]
            
            downloader = DataDownloader(
                base_url=custom_url,
                dest_folder=tmpdir,
                years_to_download=custom_years
            )
            
            assert downloader.base_url == custom_url
            assert downloader.dest_folder == tmpdir
            assert downloader.years_to_download == custom_years
    
    def test_init_creates_dest_folder(self):
        """Test that initialization creates destination folder if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "new_folder")
            assert not os.path.exists(dest)
            
            downloader = DataDownloader(dest_folder=dest)
            
            assert os.path.exists(dest)
    
    @patch('src.training_api.data.downloader.requests.get')
    def test_download_success(self, mock_get):
        """Test successful download of files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.iter_content = Mock(return_value=[b'test data'])
            mock_get.return_value = mock_response
            
            downloader = DataDownloader(
                dest_folder=tmpdir,
                years_to_download=["2010"]
            )
            downloader.download()
            
            # Check that requests.get was called 12 times (one per month)
            assert mock_get.call_count == 12
            
            # Verify URL format
            first_call_url = mock_get.call_args_list[0][0][0]
            assert "2010-01" in first_call_url
    
    @patch('src.training_api.data.downloader.requests.get')
    def test_download_creates_files(self, mock_get):
        """Test that download creates parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.iter_content = Mock(return_value=[b'test data'])
            mock_get.return_value = mock_response
            
            downloader = DataDownloader(
                dest_folder=tmpdir,
                years_to_download=["2010"]
            )
            downloader.download()
            
            # Check that files were created
            files = os.listdir(tmpdir)
            assert len(files) == 12  # 12 months
            
            # Check file naming
            assert "yellow_tripdata_2010-01.parquet" in files
            assert "yellow_tripdata_2010-12.parquet" in files
    
    @patch('src.training_api.data.downloader.requests.get')
    def test_download_handles_request_exception(self, mock_get):
        """Test handling of request exceptions."""
        import requests
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock failed request
            mock_get.side_effect = requests.exceptions.RequestException("Network error")
            
            downloader = DataDownloader(
                dest_folder=tmpdir,
                years_to_download=["2010"]
            )
            
            # Should not raise exception, just log error
            downloader.download()
            
            # No files should be created
            files = os.listdir(tmpdir)
            assert len(files) == 0
    
    @patch('src.training_api.data.downloader.requests.get')
    def test_download_handles_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        import requests
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock HTTP error
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("HTTP 404")
            mock_get.return_value = mock_response
            
            downloader = DataDownloader(
                dest_folder=tmpdir,
                years_to_download=["2010"]
            )
            
            # Should not raise exception, just log error
            downloader.download()
    
    @patch('src.training_api.data.downloader.requests.get')
    def test_download_multiple_years(self, mock_get):
        """Test downloading data for multiple years."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.iter_content = Mock(return_value=[b'test data'])
            mock_get.return_value = mock_response
            
            downloader = DataDownloader(
                dest_folder=tmpdir,
                years_to_download=["2010", "2011"]
            )
            downloader.download()
            
            # Should call requests.get 24 times (12 months * 2 years)
            assert mock_get.call_count == 24
            
            # Check files for both years
            files = os.listdir(tmpdir)
            assert "yellow_tripdata_2010-01.parquet" in files
            assert "yellow_tripdata_2011-01.parquet" in files
    
    @patch('src.training_api.data.downloader.requests.get')
    def test_download_url_format(self, mock_get):
        """Test that download constructs correct URLs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.iter_content = Mock(return_value=[b'test data'])
            mock_get.return_value = mock_response
            
            base_url = "https://example.com/data/yellow_tripdata"
            downloader = DataDownloader(
                base_url=base_url,
                dest_folder=tmpdir,
                years_to_download=["2015"]
            )
            downloader.download()
            
            # Check URL format for January
            calls = mock_get.call_args_list
            first_url = calls[0][0][0]
            assert first_url == f"{base_url}_2015-01.parquet"
            
            # Check URL format for December
            last_url = calls[11][0][0]
            assert last_url == f"{base_url}_2015-12.parquet"
    
    @patch('src.training_api.data.downloader.requests.get')
    def test_download_uses_streaming(self, mock_get):
        """Test that download uses streaming for large files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
            mock_get.return_value = mock_response
            
            downloader = DataDownloader(
                dest_folder=tmpdir,
                years_to_download=["2010"]
            )
            downloader.download()
            
            # Verify stream=True was used
            for call in mock_get.call_args_list:
                assert call[1]['stream'] is True
    
    @patch('src.training_api.data.downloader.requests.get')
    def test_download_timeout(self, mock_get):
        """Test that download uses appropriate timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.iter_content = Mock(return_value=[b'test data'])
            mock_get.return_value = mock_response
            
            downloader = DataDownloader(
                dest_folder=tmpdir,
                years_to_download=["2010"]
            )
            downloader.download()
            
            # Verify timeout was used
            for call in mock_get.call_args_list:
                assert 'timeout' in call[1]
