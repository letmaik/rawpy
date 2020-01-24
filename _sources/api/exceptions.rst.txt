Exceptions
==========

NotSupportedError
-----------------

:class:`rawpy.NotSupportedError` is raised if a feature of LibRaw is used that
was not enabled during the LibRaw library build, for example demosaic algorithms that are
part of the GPL2/GPL3 packs.

.. autoclass:: rawpy.NotSupportedError

LibRawError
-----------

:class:`rawpy.LibRawError` is the base type of :class:`rawpy.LibRawFatalError` and :class:`rawpy.LibRawNonFatalError`
and can be used to catch any LibRaw errors.

.. autoclass:: rawpy.LibRawError

LibRawFatalError
----------------

.. autoclass:: rawpy.LibRawFatalError

The following error types are subtypes of :class:`rawpy.LibRawFatalError`:

.. autoclass:: rawpy.LibRawUnsufficientMemoryError

.. autoclass:: rawpy.LibRawDataError

.. autoclass:: rawpy.LibRawIOError

.. autoclass:: rawpy.LibRawCancelledByCallbackError

.. autoclass:: rawpy.LibRawBadCropError

.. autoclass:: rawpy.LibRawTooBigError

.. autoclass:: rawpy.LibRawMemPoolOverflowError

LibRawNonFatalError
-------------------

.. autoclass:: rawpy.LibRawNonFatalError

The following error types are subtypes of :class:`rawpy.LibRawNonFatalError`:

.. autoclass:: rawpy.LibRawUnspecifiedError

.. autoclass:: rawpy.LibRawFileUnsupportedError

.. autoclass:: rawpy.LibRawRequestForNonexistentImageError

.. autoclass:: rawpy.LibRawOutOfOrderCallError

.. autoclass:: rawpy.LibRawNoThumbnailError

.. autoclass:: rawpy.LibRawUnsupportedThumbnailError

.. autoclass:: rawpy.LibRawInputClosedError

.. autoclass:: rawpy.LibRawNotImplementedError
