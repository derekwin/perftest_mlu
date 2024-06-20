/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "mlu_memory.h"
#include "perftest_parameters.h"
#include "cnrt.h"
#include MLU_PATH

#define CUCHECK(stmt) \
	do { \
	CNresult result = (stmt); \
	ASSERT(CN_SUCCESS == result); \
} while (0)

#define ACCEL_PAGE_SIZE (64 * 1024)


struct mlu_memory_ctx {
	struct memory_ctx base;
	int device_id;
	char *device_bus_id;
	CNdev cnDevice;
	CNcontext cnContext;
	bool use_dmabuf;
};


static int init_gpu(struct mlu_memory_ctx *ctx)
{
	int mlu_device_id = ctx->device_id;
	int mlu_pci_bus_id;
	int mlu_pci_device_id;
	int index;
	CNdev cn_device;

	printf("initializing MLU\n");
	CNresult error = cnInit(0);
	if (error != CN_SUCCESS) {
		printf("cnInit(0) returned %d\n", error);
		return FAILURE;
	}

	int deviceCount = 0;
	error = cnDeviceGetCount(&deviceCount);
	if (error != CN_SUCCESS) {
		printf("cnDeviceGetCount() returned %d\n", error);
		return FAILURE;
	}
	/* This function call returns 0 if there are no MLU capable devices. */
	if (deviceCount == 0) {
		printf("There are no available device(s) that support MLU\n");
		return FAILURE;
	}
	if (cn_device_id >= deviceCount) {
		fprintf(stderr, "No such device ID (%d) exists in system\n", cn_device_id);
		return FAILURE;
	}

	printf("Listing all MLU devices in system:\n");
	for (index = 0; index < deviceCount; index++) {
		CUCHECK(cnDeviceGet(&cn_device, index));
		cnDeviceGetAttribute(&mlu_pci_bus_id, CN_DEVICE_ATTRIBUTE_PCI_BUS_ID , cn_device); // 这个常量的存在性
		cnDeviceGetAttribute(&mlu_pci_device_id, CN_DEVICE_ATTRIBUTE_PCI_DEVICE_ID , cn_device);
		printf("MLU device %d: PCIe address is %02X:%02X\n", index, (unsigned int)mlu_pci_bus_id, (unsigned int)mlu_pci_device_id);
	}

	printf("\nPicking device No. %d\n", mlu_device_id);

	CUCHECK(cnDeviceGet(&ctx->cnDevice, mlu_device_id));

	char name[128];
	CUCHECK(cnDeviceGetName(name, sizeof(name), mlu_device_id));
	printf("[pid = %d, dev = %d] device name = [%s]\n", getpid(), ctx->cnDevice, name); // cuDevice
	printf("creating MLU Ctx\n");

	/* Create context */
	error = cnCtxCreate(&ctx->cnContext, CN_CTX_MAP_HOST, ctx->cnDevice);
	if (error != CN_SUCCESS) {
		printf("cnCtxCreate() error=%d\n", error);
		return FAILURE;
	}

	printf("making it the current MLU Ctx\n");
	error = cnCtxSetCurrent(ctx->cnContext);
	if (error != CN_SUCCESS) {
		printf("cnCtxSetCurrent() error=%d\n", error);
		return FAILURE;
	}

	return SUCCESS;
}
// ======================
static void free_gpu(struct mlu_memory_ctx *ctx)
{
	printf("destroying current MLU Ctx\n");
	CUCHECK(cnCtxDestroy(ctx->cnContext));
}

int mlu_memory_init(struct memory_ctx *ctx) {
	struct mlu_memory_ctx *mlu_ctx = container_of(ctx, struct mlu_memory_ctx, base);
	int return_value = 0;

	if (mlu_ctx->device_bus_id) {
		int err;

		printf("initializing MLU\n");
		CNresult error = cnInit(0);
		if (error != CN_SUCCESS) {
			printf("cnInit(0) returned %d\n", error);
			return FAILURE;
		}

		printf("Finding PCIe BUS %s\n", mlu_ctx->device_bus_id);
		err = cnDeviceGetByPCIBusId(&mlu_ctx->device_id, mlu_ctx->device_bus_id);
		if (err != 0) {
			fprintf(stderr, "cnDeviceGetByPCIBusId failed with error: %d; Failed to get PCI Bus ID (%s)\n", err, mlu_ctx->device_bus_id);
			return FAILURE;
		}
		printf("Picking GPU number %d\n", mlu_ctx->device_id);
	}

	return_value = init_gpu(mlu_ctx);
	if (return_value) {
		fprintf(stderr, "Couldn't init GPU context: %d\n", return_value);
		return FAILURE;
	}

#ifdef HAVE_MLU_DMABUF
	mlu_ctx->use_dmabuf = 0; // mlu 370 does not support dma-buf
	if (mlu_ctx->use_dmabuf) {
		// int is_supported = 0;

		// CUCHECK(cuDeviceGetAttribute(&is_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cuda_ctx->cuDevice));
		// if (!is_supported) {
		// 	fprintf(stderr, "DMA-BUF is not supported on this GPU\n");
		// 	return FAILURE;
		// }
	}
#endif

	return SUCCESS;
}

int mlu_memory_destroy(struct memory_ctx *ctx) {
	struct mlu_memory_ctx *mlu_ctx = container_of(ctx, struct mlu_memory_ctx, base);

	free_gpu(mlu_ctx);
	free(mlu_ctx);
	return SUCCESS;
}

int mlu_memory_allocate_buffer(struct memory_ctx *ctx, int alignment, uint64_t size, int *dmabuf_fd,
				uint64_t *dmabuf_offset,  void **addr, bool *can_init) {
	int error;
	size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);

	// Check if discrete or integrated GPU (tegra), for allocating memory where adequate
	struct mlu_memory_ctx *mlu_ctx = container_of(ctx, struct mlu_memory_ctx, base);
	int mlu_device_integrated;
	cnDeviceGetAttribute(&mlu_device_integrated, CN_DEVICE_ATTRIBUTE_INTEGRATED, mlu_ctx->cnDevice);
	printf("MLU device integrated: %X\n", (unsigned int)mlu_device_integrated);

	if (mlu_device_integrated == 1) {
		printf("cnMallocHost() of a %lu bytes GPU buffer\n", size);

		error = cnMallocHost(addr, buf_size);
		if (error != CN_SUCCESS) {
			printf("cnMallocHost error=%d\n", error);
			return FAILURE;
		}

		printf("allocated GPU buffer address at %p\n", addr);
		*can_init = false;
	} else {
		CNaddr d_A; // CUdeviceprt
		printf("cnMemAlloc() of a %lu bytes GPU buffer\n", size);

		error = cnMalloc(&d_A, buf_size);
		if (error != CN_SUCCESS) {
			printf("cnMemAlloc error=%d\n", error);
			return FAILURE;
		}

		printf("allocated GPU buffer address at %016llx pointer=%p\n", d_A, (void *)d_A);
		*addr = (void *)d_A;
		*can_init = false;

#ifdef HAVE_MLU_DMABUF
		{
			mlu_ctx->use_dmabuf = 0;
			if (mlu_ctx->use_dmabuf) {
				// CUdeviceptr aligned_ptr;
				// const size_t host_page_size = sysconf(_SC_PAGESIZE);
				// uint64_t offset;
				// size_t aligned_size;

				// // Round down to host page size
				// aligned_ptr = d_A & ~(host_page_size - 1);
				// offset = d_A - aligned_ptr;
				// aligned_size = (size + offset + host_page_size - 1) & ~(host_page_size - 1);

				// printf("using DMA-BUF for GPU buffer address at %#llx aligned at %#llx with aligned size %zu\n", d_A, aligned_ptr, aligned_size);
				// *dmabuf_fd = 0;
				// error = cuMemGetHandleForAddressRange((void *)dmabuf_fd, aligned_ptr, aligned_size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
				// if (error != CUDA_SUCCESS) {
				// 	printf("cuMemGetHandleForAddressRange error=%d\n", error);
				// 	return FAILURE;
				// }

				// *dmabuf_offset = offset;
			}
		}
#endif
	}

	return SUCCESS;
}

int mlu_memory_free_buffer(struct memory_ctx *ctx, int dmabuf_fd, void *addr, uint64_t size) {
	struct mlu_memory_ctx *mlu_ctx = container_of(ctx, struct mlu_memory_ctx, base);
	int mlu_device_integrated;
	cnDeviceGetAttribute(&mlu_device_integrated, CN_DEVICE_ATTRIBUTE_INTEGRATED, mlu_ctx->cnDevice);

	if (mlu_device_integrated == 1) {
		printf("deallocating GPU buffer %p\n", addr);
		cnFreeHost(addr);
	} else {
		CNaddr d_A = (CNaddr)addr;
		printf("deallocating GPU buffer %016llx\n", d_A);
		cnFree(d_A);
	}

	return SUCCESS;
}

void *mlu_memory_copy_host_buffer(void *dest, const void *src, size_t size) {
	cnMemcpy((CNaddr)dest, (CNaddr)src, size);
	return dest;
}

void *mlu_memory_copy_buffer_to_buffer(void *dest, const void *src, size_t size) {
	cuMemcpyDtoD((CNaddr)dest, (CNaddr)src, size);
	return dest;
}

bool mlu_memory_supported() {
	return true;
}

bool mlu_memory_dmabuf_supported() {
#ifdef HAVE_MLU_DMABUF
	// return true;
	return false;
#else
	return false;
#endif
}

struct memory_ctx *mlu_memory_create(struct perftest_parameters *params) {
	struct mlu_memory_ctx *ctx;

	ALLOCATE(ctx, struct mlu_memory_ctx, 1);
	ctx->base.init = mlu_memory_init;
	ctx->base.destroy = mlu_memory_destroy;
	ctx->base.allocate_buffer = mlu_memory_allocate_buffer;
	ctx->base.free_buffer = mlu_memory_free_buffer;
	ctx->base.copy_host_to_buffer = mlu_memory_copy_host_buffer;
	ctx->base.copy_buffer_to_host = mlu_memory_copy_host_buffer;
	ctx->base.copy_buffer_to_buffer = mlu_memory_copy_buffer_to_buffer;
	ctx->device_id = params->mlu_device_id;
	ctx->device_bus_id = params->mlu_device_bus_id;
	ctx->use_dmabuf = params->use_mlu_dmabuf;

	return &ctx->base;
}
