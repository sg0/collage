#pragma once
#ifndef ALLTOALLV_HPP
#define ALLTOALLV_HPP

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "utils.hpp"

template <typename T, bool LC>
class Alltoallv
{
    public:
    
    typedef typename std::conditional<LC == true, MPI_Count, int> count_t;
    typedef typename std::conditional<LC == true, MPI_Aint, int> disp_t;

    Alltoallv(MPI_Comm comm=MPI_COMM_WORLD) { SETUP(comm); }
    ~Alltoallv() { FREE(); }

    /// algorithms

    /// 1. in place
    void inplace(const T *rbuf, std::vector<count_t> const& rcounts, std::vector<disp_t> const& rdisps)
    {
        int i, size, rank, left, right, line;
        ptrdiff_t extent;
        MPI_Request req = MPI_REQUEST_NULL;
        std::vector<char> tmp_buffer;
        size_t packed_size = 0, max_size, type_size;

        /* Initialize. */
        for (i = 0, max_size = 0; i < size_; ++i) 
        {
            if (i == rank)
                continue;
            packed_size = rcounts[i] * sizeof(T);
            max_size = std::max(packed_size, max_size);
        }

        if ((1 == size) || (0 == max_size) )
            return;

        /* Allocate a temporary buffer */
        tmp_buffer.resize(max_size, 1);

        for (i = 1 ; i <= (size >> 1) ; ++i) 
        {
            right = (rank + i) % size;
            left  = (rank + size - i) % size;

            /* nothing to exchange with the peer on the right */
            if (rcounts[right] != 0) 
            {  
                MPI_Irecv((char *)rbuf + rdisps[right] * sizeof(T),
                        rcounts[right], TypeMap<T>(),
                        right, TAG_ALLTOALLV, comm_, &req);
            }

            /* exchange between left and right */
            if ((left != right) && (rcounts[left] != 0)) 
            {
                /* Send data to the left */
                MPI_Send((char *)rbuf + rdisps[left] * sizeof(T),
                        rcounts[left], TypeMap<T>(),
                        left, TAG_ALLTOALLV, comm);

                MPI_Wait(&req, MPI_STATUSES_IGNORE);

                /* Receive data from the left */
                MP_Irecv((char *)rbuf + rdisps[left] * sizeof(T),
                        rcounts[left], TypeMap<T>(),
                        left, TAG_ALLTOALLV, comm_, &req);
            }

            /* nothing to exchange with the peer on the right */
            if (rcounts[right] != 0) 
            {  
                /* Send data to the right */
                MPI_Send((char *)tmp_buffer.data(), packed_size, MPI_BYTE,
                        right, TAG_ALLTOALLV, comm_);
            }

            MPI_Wait(&req, MPI_STATUSES_IGNORE);
        }

        tmp_buffer.clear();
    }

    /// 2. pairwise
	int
		ompi_coll_base_alltoallv_intra_pairwise(const void *sbuf, ompi_count_array_t scounts, ompi_disp_array_t sdisps,
				struct ompi_datatype_t *sdtype,
				void* rbuf, ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
				struct ompi_datatype_t *rdtype,
				struct ompi_communicator_t *comm,
				mca_coll_base_module_t *module)
		{
			int line = -1, err = 0, rank, size, step = 0, sendto, recvfrom;
			size_t sdtype_size, rdtype_size;
			void *psnd, *prcv;
			ompi_request_t *req;
			ptrdiff_t sext, rext;

			if (MPI_IN_PLACE == sbuf) {
				return mca_coll_base_alltoallv_intra_basic_inplace (rbuf, rcounts, rdisps,
						rdtype, comm, module);
			}

			size = ompi_comm_size(comm);
			rank = ompi_comm_rank(comm);

			OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
						"coll:base:alltoallv_intra_pairwise rank %d", rank));

			ompi_datatype_type_size(sdtype, &sdtype_size);
			ompi_datatype_type_size(rdtype, &rdtype_size);

			ompi_datatype_type_extent(sdtype, &sext);
			ompi_datatype_type_extent(rdtype, &rext);

			/* Perform pairwise exchange starting from 1 since local exchange is done */
			for (step = 0; step < size; step++) {
				req = MPI_REQUEST_NULL;

				/* Determine sender and receiver for this step. */
				sendto  = (rank + step) % size;
				recvfrom = (rank + size - step) % size;

				/* Determine sending and receiving locations */
				psnd = (char*)sbuf + ompi_disp_array_get(sdisps, sendto) * sext;
				prcv = (char*)rbuf + ompi_disp_array_get(rdisps, recvfrom) * rext;

				/* send and receive */
				if (0 < ompi_count_array_get(rcounts, recvfrom) && 0 < rdtype_size) {
					err = MCA_PML_CALL(irecv(prcv, ompi_count_array_get(rcounts, recvfrom), rdtype, recvfrom,
								MCA_COLL_BASE_TAG_ALLTOALLV, comm, &req));
					if (MPI_SUCCESS != err) {
						line = __LINE__;
						goto err_hndl;
					}
				}

				if (0 < ompi_count_array_get(scounts, sendto) && 0 < sdtype_size) {
					err = MCA_PML_CALL(send(psnd, ompi_count_array_get(scounts, sendto), sdtype, sendto,
								MCA_COLL_BASE_TAG_ALLTOALLV, MCA_PML_BASE_SEND_STANDARD, comm));
					if (MPI_SUCCESS != err) {
						line = __LINE__;
						goto err_hndl;
					}
				}

				if (MPI_REQUEST_NULL != req) {
					err = ompi_request_wait(&req, MPI_STATUS_IGNORE);
					if (MPI_SUCCESS != err) {
						line = __LINE__;
						goto err_hndl;
					}
				}
			}

			return MPI_SUCCESS;

err_hndl:
			OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
						"%s:%4d\tError occurred %d, rank %2d at step %d", __FILE__, line,
						err, rank, step));
			(void)line;  // silence compiler warning
			return err;
		}

	/**
	 * Linear functions are copied from the basic coll module.  For
	 * some small number of nodes and/or small data sizes they are just as
	 * fast as base/tree based segmenting operations and as such may be
	 * selected by the decision functions.  These are copied into this module
	 * due to the way we select modules in V1. i.e. in V2 we will handle this
	 * differently and so will not have to duplicate code.
	 */
	int
		ompi_coll_base_alltoallv_intra_basic_linear(const void *sbuf, ompi_count_array_t scounts, ompi_disp_array_t sdisps,
				struct ompi_datatype_t *sdtype,
				void *rbuf, ompi_count_array_t rcounts, ompi_disp_array_t rdisps,
				struct ompi_datatype_t *rdtype,
				struct ompi_communicator_t *comm,
				mca_coll_base_module_t *module)
		{
			int i, size, rank, err, nreqs;
			size_t sdtype_size = 0, rdtype_size = 0;
			char *psnd, *prcv;
			ptrdiff_t sext, rext;
			ompi_request_t **preq, **reqs;
			mca_coll_base_module_t *base_module = (mca_coll_base_module_t*) module;
			mca_coll_base_comm_t *data = base_module->base_data;

			if (MPI_IN_PLACE == sbuf) {
				return  mca_coll_base_alltoallv_intra_basic_inplace (rbuf, rcounts, rdisps,
						rdtype, comm, module);
			}

			size = ompi_comm_size(comm);
			rank = ompi_comm_rank(comm);

			OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
						"coll:base:alltoallv_intra_basic_linear rank %d", rank));

			ompi_datatype_type_size(rdtype, &rdtype_size);
			ompi_datatype_type_size(sdtype, &sdtype_size);

			ompi_datatype_type_extent(sdtype, &sext);
			ompi_datatype_type_extent(rdtype, &rext);

			/* Simple optimization - handle send to self first */
			psnd = ((char *) sbuf) + ompi_disp_array_get(sdisps, rank) * sext;
			prcv = ((char *) rbuf) + ompi_disp_array_get(rdisps, rank) * rext;
			if (0 < ompi_count_array_get(scounts, rank) && 0 < sdtype_size) {
				err = ompi_datatype_sndrcv(psnd, ompi_count_array_get(scounts, rank), sdtype,
						prcv, ompi_count_array_get(rcounts, rank), rdtype);
				if (MPI_SUCCESS != err) {
					return err;
				}
			}

			/* If only one process, we're done. */
			if (1 == size) {
				return MPI_SUCCESS;
			}

			/* Now, initiate all send/recv to/from others. */
			nreqs = 0;
			reqs = preq = ompi_coll_base_comm_get_reqs(data, 2 * size);
			if( NULL == reqs ) { err = OMPI_ERR_OUT_OF_RESOURCE; goto err_hndl; }

			/* Post all receives first */
			for (i = 0; i < size; ++i) {
				if (i == rank) {
					continue;
				}

				if (0 < ompi_count_array_get(rcounts, i) && 0 < rdtype_size) {
					++nreqs;
					prcv = ((char *) rbuf) + ompi_disp_array_get(rdisps, i) * rext;
					err = MCA_PML_CALL(irecv_init(prcv, ompi_count_array_get(rcounts, i), rdtype,
								i, MCA_COLL_BASE_TAG_ALLTOALLV, comm,
								preq++));
					if (MPI_SUCCESS != err) { goto err_hndl; }
				}
			}

			/* Now post all sends */
			for (i = 0; i < size; ++i) {
				if (i == rank) {
					continue;
				}

				if (0 < ompi_count_array_get(scounts, i) && 0 < sdtype_size) {
					++nreqs;
					psnd = ((char *) sbuf) + ompi_disp_array_get(sdisps, i) * sext;
					err = MCA_PML_CALL(isend_init(psnd, ompi_count_array_get(scounts, i), sdtype,
								i, MCA_COLL_BASE_TAG_ALLTOALLV,
								MCA_PML_BASE_SEND_STANDARD, comm,
								preq++));
					if (MPI_SUCCESS != err) { goto err_hndl; }
				}
			}

			/* Start your engines.  This will never return an error. */
			MCA_PML_CALL(start(nreqs, reqs));

			/* Wait for them all.  If there's an error, note that we don't care
			 * what the error was -- just that there *was* an error.  The PML
			 * will finish all requests, even if one or more of them fail.
			 * i.e., by the end of this call, all the requests are free-able.
			 * So free them anyway -- even if there was an error, and return the
			 * error after we free everything. */
			err = ompi_request_wait_all(nreqs, reqs, MPI_STATUSES_IGNORE);

err_hndl:
			/* find a real error code */
			if (MPI_ERR_IN_STATUS == err) {
				for( i = 0; i < nreqs; i++ ) {
					if (MPI_REQUEST_NULL == reqs[i]) continue;
					if (MPI_ERR_PENDING == reqs[i]->req_status.MPI_ERROR) continue;
					if (reqs[i]->req_status.MPI_ERROR != MPI_SUCCESS) {
						err = reqs[i]->req_status.MPI_ERROR;
						break;
					}
				}
			}
			/* Free the requests in all cases as they are persistent */
			ompi_coll_base_free_reqs(reqs, nreqs);

			return err;
		}

    private:
};

#endif
