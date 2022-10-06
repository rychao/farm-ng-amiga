from pathlib import Path

import pytest

from farm_ng.core import uri_pb2
from farm_ng.core.events_file_reader import EventsFileReader
from farm_ng.core.events_file_writer import EventsFileWriter
from farm_ng.oak import oak_pb2


@pytest.fixture(name="writer")
def fixture_writer(tmpdir) -> EventsFileWriter:
    file_name = Path(tmpdir) / "event.log"
    return EventsFileWriter(file_name)


@pytest.fixture(name="reader")
def fixture_reader(tmpdir) -> EventsFileReader:
    file_name = Path(tmpdir) / "event.log"
    return EventsFileReader(file_name)


class TestEventsWriter:
    def test_smoke(self, writer: EventsFileWriter) -> None:
        # empty object
        assert writer.is_closed()
        assert not writer.is_open()
        assert writer.file_name is None

    def test_open_close(self, writer: EventsFileWriter) -> None:
        # open the file
        assert writer.open()
        assert not writer.is_closed()
        assert writer.is_open()
        assert writer.file_name.name == "event.log"  # type: ignore
        # close the file
        assert writer.close()
        assert writer.is_closed()
        assert not writer.is_open()
        assert writer.file_name is None

    def test_write_images(self, writer: EventsFileWriter) -> None:
        assert writer.open()
        uri = uri_pb2.Uri()
        frame = oak_pb2.OakFrame()
        frame.image_data = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9])
        writer.write(frame, uri)
        assert writer.close()


class TestEventsReader:
    def test_smoke(self, reader: EventsFileReader) -> None:
        # empty object
        assert reader.is_closed()
        assert not reader.is_open()
        assert reader.file_name is None

    def test_open_close(self, writer: EventsFileWriter, reader: EventsFileReader) -> None:
        # touch file
        assert writer.open()
        assert writer.close()
        # open the file
        assert reader.open()
        assert not reader.is_closed()
        assert reader.is_open()
        assert reader.file_name.name == "event.log"  # type: ignore
        # close the file
        assert reader.close()
        assert reader.is_closed()
        assert not reader.is_open()
        assert reader.file_name is None

    def test_write_read(self, writer: EventsFileWriter, reader: EventsFileReader) -> None:
        # write file
        assert writer.open()
        image_data = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9])
        uri = uri_pb2.Uri(scheme="farm_ng.oak.oak_pb2", authority="OakFrame")
        frame = oak_pb2.OakFrame(image_data=image_data)
        writer.write(frame, uri)
        assert writer.close()
        # read back the data
        assert reader.open()
        _, frame_out = reader.read()  # type: ignore
        assert reader.close()
        assert frame_out.image_data == frame.image_data
